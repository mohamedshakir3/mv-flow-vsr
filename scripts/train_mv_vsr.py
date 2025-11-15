import os, glob, random, argparse, yaml
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
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
    REDS GT + codec-aware LR data (LR + bidirectional MVs + residuals).

    Layout:
      reds_root/
        train_sharp/<clip>/*.png   (GT)
        val_sharp/<clip>/*.png
      codec_root/
        <clip>/
          lr/*.png
          mv_fwd/*.npz   (flow_fwd: (2,H,W), for frame index t)
          mv_bwd/*.npz   (flow_bwd: (2,H,W), for frame index t)
          residual/*.npy (residual for frame index t, t>0)
    """
    def __init__(self, reds_root, codec_root, split,
                 scale=4, seq_len=5, crop_lr=None, augment=False,
                 img_tmpl="{:08d}.png"):
        assert split in ("train", "val", "test")
        self.scale = scale
        self.seq_len = seq_len
        self.crop_lr = crop_lr
        self.augment = augment
        self.img_tmpl = img_tmpl

        self.gt_root = os.path.join(reds_root, f"{split}_sharp")
        self.codec_root = codec_root

        self.clips = sorted([
            d for d in os.listdir(self.codec_root)
            if os.path.isdir(os.path.join(self.codec_root, d))
        ])

        self.samples = []
        for clip in self.clips:
            lr_dir = os.path.join(self.codec_root, clip, "lr")
            T = len(glob.glob(os.path.join(lr_dir, "*.png")))
            if T >= seq_len:
                for s in range(0, T - seq_len + 1):
                    self.samples.append((clip, s))

        if not self.samples:
            raise RuntimeError("No samples found. Check codec_root and seq_len.")

    def _load_seq(self, clip, start):
        """
        Load LR, GT, mv_fwd, mv_bwd, residual for a window [start, start+seq_len).
        """
        lr_dir   = os.path.join(self.codec_root, clip, "lr")
        mvf_dir  = os.path.join(self.codec_root, clip, "mv_fwd")
        mvb_dir  = os.path.join(self.codec_root, clip, "mv_bwd")
        res_dir  = os.path.join(self.codec_root, clip, "residual")
        gt_dir   = os.path.join(self.gt_root, clip)

        imgs_lr, imgs_gt = [], []
        mv_fwd_list, mv_bwd_list, res_list = [], [], []

        for t in range(start, start + self.seq_len):
            fn = self.img_tmpl.format(t)
            # LR frame
            lr_path = os.path.join(lr_dir, fn)
            imgs_lr.append(imread_rgb(lr_path))
            # GT frame
            gt_path = os.path.join(gt_dir, fn)
            imgs_gt.append(imread_rgb(gt_path))

        lr_arr = np.stack(imgs_lr, axis=0)  # (T,H,W,3)
        gt_arr = np.stack(imgs_gt, axis=0)  # (T,4H,4W,3)
        T, H, W, _ = lr_arr.shape

        for t in range(start, start + self.seq_len):
            base = self.img_tmpl.format(t)
            stem = os.path.splitext(base)[0]  # "00000010"

            if t == 0:
                mvf = np.zeros((2, H, W), np.float32)
            else:
                p_fwd = os.path.join(mvf_dir, f"{stem}_mv_fwd.npz")
                if os.path.exists(p_fwd):
                    f = np.load(p_fwd)["flow_fwd"].astype(np.float32)  # (2,H,W) or (H,W,2)
                    if f.ndim == 3 and f.shape[0] != 2:
                        # if stored as (H,W,2), transpose
                        f = np.transpose(f, (2, 0, 1))
                    mvf = f
                else:
                    mvf = np.zeros((2, H, W), np.float32)
            mv_fwd_list.append(mvf)

            p_bwd = os.path.join(mvb_dir, f"{stem}_mv_bwd.npz")
            if os.path.exists(p_bwd):
                b = np.load(p_bwd)["flow_bwd"].astype(np.float32)
                if b.ndim == 3 and b.shape[0] != 2:
                    b = np.transpose(b, (2, 0, 1))
                mvb = b
            else:
                mvb = np.zeros((2, H, W), np.float32)
            mv_bwd_list.append(mvb)

            if t == 0:
                res = np.zeros((H, W), np.float32)
            else:
                p_res = os.path.join(res_dir, f"{stem}_res.npy")
                if os.path.exists(p_res):
                    res = np.load(p_res).astype(np.float32)  # (H,W)
                else:
                    res = np.zeros((H, W), np.float32)
            res_list.append(res)

        mv_fwd_arr = np.stack(mv_fwd_list, axis=0)  # (T,2,H,W)
        mv_bwd_arr = np.stack(mv_bwd_list, axis=0)  # (T,2,H,W)
        res_arr    = np.stack(res_list,    axis=0)  # (T,H,W)

        return lr_arr, gt_arr, mv_fwd_arr, mv_bwd_arr, res_arr

    def _random_crop(self, lr, gt, mv_fwd, mv_bwd, res):
        T, H, W, _ = lr.shape
        ch = cw = self.crop_lr
        if ch is None or H < ch or W < cw:
            return lr, gt, mv_fwd, mv_bwd, res
        y0 = random.randint(0, H - ch)
        x0 = random.randint(0, W - cw)

        lr = lr[:, y0:y0+ch, x0:x0+cw, :]
        gt = gt[:, y0*self.scale:(y0+ch)*self.scale,
                   x0*self.scale:(x0+cw)*self.scale, :]

        mv_fwd = mv_fwd[:, :, y0:y0+ch, x0:x0+cw]
        mv_bwd = mv_bwd[:, :, y0:y0+ch, x0:x0+cw]
        res    = res[:, y0:y0+ch, x0:x0+cw]

        return lr, gt, mv_fwd, mv_bwd, res

    def _augment(self, lr, gt, mv_fwd, mv_bwd, res):
        if random.random() < 0.5:
            lr  = lr[:, :, ::-1]
            gt  = gt[:, :, ::-1]
            res = res[:, :, ::-1]
            mv_fwd = mv_fwd[:, :, :, ::-1]
            mv_bwd = mv_bwd[:, :, :, ::-1]
            mv_fwd[:, 0] *= -1.0
            mv_bwd[:, 0] *= -1.0

        if random.random() < 0.5:
            lr  = lr[:, ::-1, :]
            gt  = gt[:, ::-1, :]
            res = res[:, ::-1, :]
            mv_fwd = mv_fwd[:, :, ::-1, :]
            mv_bwd = mv_bwd[:, :, ::-1, :]
            mv_fwd[:, 1] *= -1.0
            mv_bwd[:, 1] *= -1.0

        if random.random() < 0.5:
            lr  = lr.transpose(0, 2, 1, 3)
            gt  = gt.transpose(0, 2, 1, 3)
            res = res.transpose(0, 2, 1)
            mv_fwd = mv_fwd.transpose(0, 1, 3, 2)
            mv_bwd = mv_bwd.transpose(0, 1, 3, 2)
            mv_fwd = mv_fwd[:, [1, 0]]
            mv_bwd = mv_bwd[:, [1, 0]]

        return lr, gt, mv_fwd, mv_bwd, res

    def __getitem__(self, idx):
        clip, s = self.samples[idx]
        lr, gt, mv_fwd, mv_bwd, res = self._load_seq(clip, s)

        if self.crop_lr is not None:
            lr, gt, mv_fwd, mv_bwd, res = self._random_crop(lr, gt, mv_fwd, mv_bwd, res)
        if self.augment:
            lr, gt, mv_fwd, mv_bwd, res = self._augment(lr, gt, mv_fwd, mv_bwd, res)

        imgs = torch.stack([to_tensor01(im) for im in lr], dim=0)  # (T,3,H,W)
        gts  = torch.stack([to_tensor01(im) for im in gt], dim=0)  # (T,3,4H,4W)

        mv_fwd_t = torch.from_numpy(np.ascontiguousarray(mv_fwd)).float()  # (T,2,H,W)
        mv_bwd_t = torch.from_numpy(np.ascontiguousarray(mv_bwd)).float()  # (T,2,H,W)
        res_t    = torch.from_numpy(np.ascontiguousarray(res)).unsqueeze(1).float()  # (T,1,H,W)

        return imgs, gts, mv_fwd_t, mv_bwd_t, res_t, clip, s

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
    
class CodecFlowHead(nn.Module):
    """
    Predicts per-pixel from codec features:
      - forward MV
      - backward MV
      - residual map
    """
    def __init__(self, hidden: int = 64):
        super().__init__()
        # mv_fwd: 2ch, mv_bwd: 2ch, residual: 1ch  => 5 total
        in_ch = 5
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, 3, padding=1),  # output flow dx,dy
        )

    def forward(self, mv_fwd, mv_bwd, residual):
        # mv_*: (B, 2, H, W), residual: (B, 1, H, W) or (B, H, W)
        if residual.dim() == 3:
            residual = residual.unsqueeze(1)
        x = torch.cat([mv_fwd, mv_bwd, residual], dim=1)
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

        # LR feature extractor
        body = [nn.Conv2d(3, mid, 3, 1, 1), nn.ReLU(inplace=False)]
        for _ in range(blocks):
            body += [ResidualBlock(mid)]
        self.feat_ex = nn.Sequential(*body)

        # motion + fusion + upsampling
        self.mvwarp = MVWarp(align_corners=True)
        self.codec_flow = CodecFlowHead(hidden=64)
        self.fuse = nn.Sequential(
            nn.Conv2d(mid * 2 + 1, mid, 3, 1, 1),  # feat_t + warped_state + conf
            nn.ReLU(inplace=False),
            nn.Conv2d(mid, mid, 3, 1, 1),
        )
        self.up = nn.Sequential(
            nn.Conv2d(mid, mid * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid, mid * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid, 3, 3, 1, 1),
        )

    def step(self, lr_t, mv_fwd_t, mv_bwd_t, res_t, state):
        """
        lr_t:    (B, 3, H, W)
        mv_*_t:  (B, 2, H, W)
        res_t:   (B, 1, H, W) or (B, H, W)
        state:   (B, mid, H, W) or None
        """
        # current frame features
        feat_t = self.feat_ex(lr_t)  # (B, mid, H, W)

        # first frame: no motion info, init state from frame itself
        if state is None:
            state = feat_t
            sr_t = self.up(state)
            return sr_t, state

        if res_t.dim() == 3:
            res_t = res_t.unsqueeze(1)  # (B,1,H,W)

        # 1) codec-based flow from MV_fwd, MV_bwd, residual
        F_codec = self.codec_flow(mv_fwd_t, mv_bwd_t, res_t)  # (B,2,H,W)

        # 2) warp previous state
        # warped: (B,mid,H,W), conf: (B,1,H,W)
        warped, conf = self.mvwarp(state, F_codec)

        # 3) fuse warped state + current features + confidence
        x = torch.cat([feat_t, warped, conf], dim=1)          # (B,2*mid+1,H,W)
        state = self.fuse(x)

        # 4) reconstruct SR
        sr_t = self.up(state)
        return sr_t, state

    def forward(self, imgs, mv_fwd, mv_bwd, residual):
        """
        imgs:     (B,T,3,H,W)
        mv_fwd:   (B,T,2,H,W)
        mv_bwd:   (B,T,2,H,W)
        residual: (B,T,1,H,W)
        """
        B, T, _, H, W = imgs.shape
        outs = []
        state = None

        for t in range(T):
            lr_t  = imgs[:, t]       # (B,3,H,W)
            mvf_t = mv_fwd[:, t]     # (B,2,H,W)
            mvb_t = mv_bwd[:, t]     # (B,2,H,W)
            res_t = residual[:, t]   # (B,1,H,W)

            sr_t, state = self.step(lr_t, mvf_t, mvb_t, res_t, state)
            outs.append(sr_t)

        return torch.stack(outs, dim=1)  # (B,T,3,4H,4W)

# ------------------------ Train / Val ------------------------
@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=torch.float16):
    model.eval()
    psnr_sum, ncount = 0.0, 0
    for imgs, gts, mv_fwd, mv_bwd, residual, _, _ in tqdm(loader, desc="val", leave=False):
        imgs     = imgs.to(device, non_blocking=True)       # (B,T,3,H,W)
        gts      = gts.to(device, non_blocking=True)        # (B,T,3,4H,4W)
        mv_fwd   = mv_fwd.to(device, non_blocking=True)     # (B,T,2,H,W)
        mv_bwd   = mv_bwd.to(device, non_blocking=True)     # (B,T,2,H,W)
        residual = residual.to(device, non_blocking=True)   # (B,T,1,H,W)

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            sr = model(imgs, mv_fwd, mv_bwd, residual)

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

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Datasets / Loaders
    train_ds = RedsMVSRDataset(
            reds_root=args.reds_root,
            codec_root=args.flows_root,
            split="train",
            scale=args.scale,
            seq_len=args.seq_len,
            crop_lr=args.crop_lr,
            augment=True,
            img_tmpl=args.img_tmpl,
    )
    val_ds = RedsMVSRDataset(
        reds_root=args.val_reds_root,
        codec_root=args.val_flows_root,
        split="val",
        scale=args.scale,
        seq_len=args.seq_len,
        crop_lr=None,
        augment=False,
        img_tmpl=args.img_tmpl,
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
    
    writer = SummaryWriter(log_dir=args.out_dir)
    tb_step = 0


    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}")
        for imgs, gts, mv_fwd, mv_bwd, residual, _, _ in pbar:
            imgs     = imgs.to(device, non_blocking=True)
            gts      = gts.to(device, non_blocking=True)
            mv_fwd   = mv_fwd.to(device, non_blocking=True)
            mv_bwd   = mv_bwd.to(device, non_blocking=True)
            residual = residual.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                sr   = model(imgs, mv_fwd, mv_bwd, residual)  # (B,T,3,4H,4W)
                loss = charbonnier_loss(sr, gts)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix(loss=f"{float(loss.item()):.4f}")
            
            writer.add_scalar("train/loss", loss.item(), tb_step)
            tb_step += 1

        val_psnr = evaluate(model, val_loader, device, amp_dtype=amp_dtype)
        print(f"[Epoch: {epoch}] Validation PSNR: {val_psnr:.3f} dB")

        writer.add_scalar("val/psnr", val_psnr, epoch)

        ckpt = os.path.join(args.out_dir, f"epoch_{epoch:04d}.pth")
        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict()}, ckpt)
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pth"))
            print(f"Best PSNR: ({best_psnr:.3f} dB) saved.")

if __name__ == "__main__":
    main()
