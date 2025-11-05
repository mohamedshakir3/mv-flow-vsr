import os, glob, random, argparse, yaml
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ------------------------ Utils ------------------------
def imread_rgb(p):
    bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def to_tensor01(img):
    if not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

@torch.no_grad()
def psnr_torch_perframe(x, y, eps=1e-12):
    """x,y: (B,T,C,H,W) -> mean PSNR over (B,T)"""
    mse = ((x - y) ** 2).mean(dim=(2, 3, 4)).clamp_min(eps)
    return (10.0 * torch.log10(1.0 / mse)).mean()

def charbonnier_loss(x, y, eps=1e-3):
    return torch.mean(torch.sqrt((x - y) ** 2 + eps ** 2))

# ------------------------ Dataset ------------------------
class RedsMVSRDataset(Dataset):
    """
    REDS + precomputed backward flows (t <- t-1) at LR resolution.
    Layout:
      reds_root/
        train_sharp_bicubic/X4/<clip>/*.png   (LR)
        train_sharp/<clip>/*.png               (GT)
      flows_root/
        <clip>/<t:06d>_mv.npz  (contains 'flow_bwd': HxWx2, dx,dy) for t>=1
    """
    def __init__(self, reds_root, split, flows_root, scale=4, seq_len=5,
                 crop_lr=None, augment=False,
                 img_tmpl="{:08d}.png", flow_tmpl="{t:06d}_mv.npz"):
        assert split in ("train", "val", "test")
        self.scale = scale
        self.seq_len = seq_len
        self.crop_lr = crop_lr
        self.augment = augment
        self.img_tmpl = img_tmpl
        self.flow_tmpl = flow_tmpl
        self.lr_root = os.path.join(reds_root, f"{split}_sharp_bicubic/X{scale}")
        self.gt_root = os.path.join(reds_root, f"{split}_sharp")
        self.flows_root = flows_root

        self.clips = sorted([d for d in os.listdir(self.lr_root)
                             if os.path.isdir(os.path.join(self.lr_root, d))])
        self.samples = []
        for clip in self.clips:
            T = len(glob.glob(os.path.join(self.lr_root, clip, "*.png")))
            if T >= seq_len:
                for s in range(0, T - seq_len + 1):
                    self.samples.append((clip, s))
        if not self.samples:
            raise RuntimeError("No samples found. Check paths and seq_len.")

    def _load_seq(self, clip, start):
        lr_dir = os.path.join(self.lr_root, clip)
        gt_dir = os.path.join(self.gt_root, clip)
        imgs_lr, imgs_gt = [], []
        for t in range(start, start + self.seq_len):
            fn = self.img_tmpl.format(t)
            imgs_lr.append(imread_rgb(os.path.join(lr_dir, fn)))
            imgs_gt.append(imread_rgb(os.path.join(gt_dir, fn)))
        lr_arr = np.stack(imgs_lr, axis=0)  # T,H,W,3
        gt_arr = np.stack(imgs_gt, axis=0)  # T,4H,4W,3

        flows = []
        flow_dir = os.path.join(self.flows_root, clip)
        H, W = lr_arr.shape[1:3]
        for t in range(start + 1, start + self.seq_len):
            p = os.path.join(flow_dir, self.flow_tmpl.format(t=t))
            if os.path.exists(p):
                f = np.load(p)["flow_bwd"].astype(np.float32)  # H,W,2
            else:
                f = np.zeros((H, W, 2), np.float32)
            flows.append(f)
        flow_arr = np.stack(flows, axis=0)  # T-1,H,W,2
        return lr_arr, gt_arr, flow_arr

    def _random_crop(self, lr, gt, flows):
        T, H, W, _ = lr.shape
        ch = cw = self.crop_lr
        if ch is None or H < ch or W < cw:
            return lr, gt, flows
        y0 = random.randint(0, H - ch)
        x0 = random.randint(0, W - cw)
        lr = lr[:, y0:y0 + ch, x0:x0 + cw, :]
        gt = gt[:, y0 * self.scale:(y0 + ch) * self.scale,
                x0 * self.scale:(x0 + cw) * self.scale, :]
        flows = flows[:, y0:y0 + ch, x0:x0 + cw, :]
        return lr, gt, flows

    def _augment(self, lr, gt, flows):
        if random.random() < 0.5:
            lr = lr[:, :, ::-1]; gt = gt[:, :, ::-1]; flows = flows[:, :, ::-1]
            flows[..., 0] *= -1 
        if random.random() < 0.5:
            lr = lr[:, ::-1, :]; gt = gt[:, ::-1, :]; flows = flows[:, ::-1, :]
            flows[..., 1] *= -1
        if random.random() < 0.5:
            lr = lr.transpose(0, 2, 1, 3); gt = gt.transpose(0, 2, 1, 3)
            flows = flows.transpose(0, 2, 1, 3); flows = flows[..., [1, 0]]
        return lr, gt, flows

    def __getitem__(self, idx):
        clip, s = self.samples[idx]
        lr, gt, flows = self._load_seq(clip, s)
        if self.crop_lr is not None:
            lr, gt, flows = self._random_crop(lr, gt, flows)
        if self.augment:
            lr, gt, flows = self._augment(lr, gt, flows)
        imgs = torch.stack([to_tensor01(im) for im in lr], dim=0)        # T,3,H,W
        gts  = torch.stack([to_tensor01(im) for im in gt], dim=0)        # T,3,4H,4W
        fls  = torch.from_numpy(np.ascontiguousarray(flows)).permute(0, 3, 1, 2).float()  # T-1,2,H,W
        return imgs, gts, fls, clip, s

    def __len__(self):
        return len(self.samples)

# ------------------------ Model ------------------------
class ResidualBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1), nn.ReLU(inplace=False),
            nn.Conv2d(nf, nf, 3, 1, 1)
        )
    def forward(self, x): return x + self.body(x)

class ResidualFlowHead(nn.Module):
    """Predict ΔF at feature resolution"""
    def __init__(self, c_feat=64, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_feat + c_feat + 2, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, 3, padding=1)
        )

    def forward(self, feat_tm1, feat_t, flow_in):
        # All at LR/feature resolution; flow_in in LR pixel units
        x = torch.cat([feat_tm1, feat_t, flow_in], dim=1)
        return self.net(x)
class MVWarp(nn.Module):
    def __init__(self, align_corners=True):
        super().__init__()
        self.align_corners = align_corners
        self.register_buffer("_base", None, persistent=False)  # (1,H,W,2)

    def _grid(self, H, W, device):
        g = self._base
        if g is None or g.shape[1] != H or g.shape[2] != W or g.device != device:
            xs = torch.linspace(-1, 1, W, device=device)
            ys = torch.linspace(-1, 1, H, device=device)
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            g = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # (1,H,W,2)
            self._base = g
        return g

    def forward(self, feat_tm1, flow_pix):
        # feat_tm1: (B,C,H,W) ; flow_pix: (B,2,H,W) backward t<-t-1 in pixels
        B, C, H, W = feat_tm1.shape
        base = self._grid(H, W, feat_tm1.device)
        sx = 2.0 / max(W - 1, 1); sy = 2.0 / max(H - 1, 1)
        nx = flow_pix[:, 0] * sx; ny = flow_pix[:, 1] * sy
        grid = torch.stack([base[..., 0] + nx, base[..., 1] + ny], dim=-1)  # (B,H,W,2)
        warped = F.grid_sample(feat_tm1, grid, mode='bilinear',
                               padding_mode="zeros", align_corners=self.align_corners)
        valid = ((grid[..., 0] > -1) & (grid[..., 0] < 1) &
                 (grid[..., 1] > -1) & (grid[..., 1] < 1)).unsqueeze(1).float()
        mag = torch.linalg.vector_norm(flow_pix, dim=1, keepdim=True)
        conf = valid * torch.exp(-(mag / 8.0) ** 2)
        return warped, conf

class MVSR(nn.Module):
    def __init__(self, mid=64, blocks=15, scale=4):
        super().__init__()
        self.scale = scale
        body = [nn.Conv2d(3, mid, 3, 1, 1), nn.ReLU(inplace=False)]
        for _ in range(blocks): body += [ResidualBlock(mid)]
        self.feat_ex = nn.Sequential(*body)
        self.mvwarp = MVWarp(align_corners=True)
        self.refine = ResidualFlowHead(c_feat=mid, hidden=64)
        self.fuse   = nn.Sequential(
            nn.Conv2d(mid * 2 + 1, mid, 3, 1, 1), nn.ReLU(inplace=False),
            nn.Conv2d(mid, mid, 3, 1, 1)
        )
        self.up = nn.Sequential(
            nn.Conv2d(mid, mid * 4, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(inplace=False),
            nn.Conv2d(mid, mid * 4, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(inplace=False),
            nn.Conv2d(mid, 3, 3, 1, 1)
        )

    def step(self, img_t, flow_t, state):
        cur = self.feat_ex(img_t)  # LR feature space
        if state is None or flow_t is None:
            new_state = cur
        else:
            delta = self.refine(state, cur, flow_t)
            flow_t = flow_t + delta
            prev_w, conf = self.mvwarp(state, flow_t)
            new_state = self.fuse(torch.cat([cur, prev_w, conf], dim=1))
        sr_t = self.up(new_state)
        return sr_t, new_state

    def forward(self, imgs, flows):
        """imgs: (B,T,3,H,W), flows: (B,T-1,2,H,W)"""
        B, T, _, H, W = imgs.shape
        outs, state = [], None
        for t in range(T):
            flow_t = None if t == 0 else flows[:, t - 1]
            sr_t, state = self.step(imgs[:, t], flow_t, state)
            outs.append(sr_t)
        return torch.stack(outs, dim=1)  # (B,T,3,4H,4W)

# ------------------------ Train / Val ------------------------
@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=torch.float16):
    model.eval()
    psnr_sum, ncount = 0.0, 0
    for imgs, gts, flows, _, _ in tqdm(loader, desc="val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        gts  = gts.to(device, non_blocking=True)
        flows= flows.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            sr = model(imgs, flows)
        psnr_sum += float(psnr_torch_perframe(sr, gts)) * imgs.size(0)
        ncount   += imgs.size(0)
    model.train()
    return psnr_sum / max(ncount, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default=None, help="YAML config file")
    ap.add_argument("--num_workers", type=int, default=32)
    ap.add_argument("--flow_tmpl", default="{t:06d}_mv.npz")
    ap.add_argument("--img_tmpl",  default="{:08d}.png")
    ap.add_argument("--out_dir",   default="out_mvsr")
    args = ap.parse_args()
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f) or {}

    def apply_dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                apply_dict(v)
            else:
                if hasattr(args, k):
                    setattr(args, k, v)

    apply_dict(cfg)

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Datasets / Loaders
    train_ds = RedsMVSRDataset(
        reds_root=args.reds_root, split="train",
        flows_root=args.flows_root, scale=args.scale,
        seq_len=args.seq_len, crop_lr=args.crop_lr, augment=True,
        img_tmpl=args.img_tmpl, flow_tmpl=args.flow_tmpl
    )
    val_ds = RedsMVSRDataset(
        reds_root=args.val_reds_root, split="val",
        flows_root=args.val_flows_root, scale=args.scale,
        seq_len=args.seq_len, crop_lr=None, augment=False,
        img_tmpl=args.img_tmpl, flow_tmpl=args.flow_tmpl
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(1, min(2, args.num_workers)), pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )

    model = MVSR(mid=64, blocks=15, scale=args.scale).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler()

    os.makedirs(args.out_dir, exist_ok=True)
    best_psnr = -1.0
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}")
        for imgs, gts, flows, _, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            gts  = gts.to(device, non_blocking=True)
            flows= flows.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                sr = model(imgs, flows)                  # (B,T,3,4H,4W)
                loss = charbonnier_loss(sr, gts)         # whole sequence loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix(loss=f"{float(loss.item()):.4f}")

        val_psnr = evaluate(model, val_loader, device, amp_dtype=amp_dtype)
        print(f"[Epoch: {epoch}] Validation PSNR: {val_psnr:.3f} dB")

        ckpt = os.path.join(args.out_dir, f"epoch_{epoch:04d}.pth")
        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict()}, ckpt)
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pth"))
            print(f"Best PSNR: ({best_psnr:.3f} dB) saved.")

if __name__ == "__main__":
    main()
