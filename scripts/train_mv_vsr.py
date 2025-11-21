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
        part_dir = os.path.join(self.codec_root, clip, "partition_maps")
        

        imgs_lr, imgs_gt = [], []
        mv_fwd_list, mv_bwd_list, res_list, part_list = [], [], [], []

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
            
            p_part = os.path.join(part_dir, f"{stem}_part.npy")
            if os.path.exists(p_part):
                pmap = np.load(p_part).astype(np.int64) 
            else:
                pmap = np.full((H//16, W//16), 2, dtype=np.int64)
            part_list.append(pmap)

            if t == 0:
                res = np.zeros((H, W), np.float32)
            else:
                p_res = os.path.join(res_dir, f"{stem}_res.npy")
                if os.path.exists(p_res):
                    res = np.load(p_res).astype(np.float32)  # (H,W)
                else:
                    res = np.zeros((H, W), np.float32)
            res_list.append(res)

        part_arr = np.stack(part_list, axis=0) # (T, H/16, W/16)
        mv_fwd_arr = np.stack(mv_fwd_list, axis=0)  # (T,2,H,W)
        mv_bwd_arr = np.stack(mv_bwd_list, axis=0)  # (T,2,H,W)
        res_arr    = np.stack(res_list,    axis=0)  # (T,H,W)

        return lr_arr, gt_arr, mv_fwd_arr, mv_bwd_arr, res_arr, part_arr

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
        lr, gt, mv_fwd, mv_bwd, res, part = self._load_seq(clip, s)

        if self.crop_lr is not None:
            lr, gt, mv_fwd, mv_bwd, res = self._random_crop(lr, gt, mv_fwd, mv_bwd, res)
        if self.augment:
            lr, gt, mv_fwd, mv_bwd, res = self._augment(lr, gt, mv_fwd, mv_bwd, res)

        imgs = torch.stack([to_tensor01(im) for im in lr], dim=0)  # (T,3,H,W)
        gts  = torch.stack([to_tensor01(im) for im in gt], dim=0)  # (T,3,4H,4W)

        mv_fwd_t = torch.from_numpy(np.ascontiguousarray(mv_fwd)).float()  # (T,2,H,W)
        mv_bwd_t = torch.from_numpy(np.ascontiguousarray(mv_bwd)).float()  # (T,2,H,W)
        res_t    = torch.from_numpy(np.ascontiguousarray(res)).unsqueeze(1).float()  # (T,1,H,W)
        part_t = torch.from_numpy(np.ascontiguousarray(part)).long() # (T, H/16, W/16)
        return imgs, gts, mv_fwd_t, mv_bwd_t, res_t, part_t, clip, s

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
    Predicts a residual flow correction (delta_flow)
    from a MV prior and a residual map.
    """
    def __init__(self, hidden: int = 64):
        super().__init__()
        # mv: 2ch, residual: 1ch  => 3 total
        in_ch = 3
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, 3, padding=1),  # output flow dx,dy
        )

    def forward(self, mv, residual):
        # mv: (B, 2, H, W), residual: (B, 1, H, W) or (B, H, W)
        if residual.dim() == 3:
            residual = residual.unsqueeze(1)
        x = torch.cat([mv, residual], dim=1)
        return self.net(x)

class MVRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, 1, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, mv):
        noise = self.net(mv)
        return mv + noise

class SFTResidualBlock(nn.Module):
    """
    Spatial Feature Transform (SFT) Residual Block.
    Uses the 'cond' feature map to modulate the main features.
    """
    def __init__(self, nf):
        super().__init__()
        self.sft_scale = nn.Conv2d(nf, nf, 1)
        self.sft_shift = nn.Conv2d(nf, nf, 1)
        
        nn.init.zeros_(self.sft_scale.weight)
        nn.init.zeros_(self.sft_scale.bias)
        nn.init.zeros_(self.sft_shift.weight)
        nn.init.zeros_(self.sft_shift.bias)
        
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1)
        )

    def forward(self, x, cond):
        scale = self.sft_scale(cond)
        shift = self.sft_shift(cond)
        
        modulated_x = x * (1 + scale) + shift
        
        return x + self.body(modulated_x)
    
class PartitionMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_large = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
        
        self.conv_inter = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
        
        self.conv_small = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )

    def forward(self, x, partition_map):
        """
        x: (B, C, H, W)
        partition_map: (B, H_map, W_map) Integer tensor {0, 1, 2}
        """
        B, C, H, W = x.shape
        
        if partition_map.shape[-1] != W:
            partition_map = F.interpolate(
                partition_map.unsqueeze(1).float(), 
                size=(H, W), 
                mode='nearest'
            ).squeeze(1).long()
            
        # mask_large: 1 where map==0 (16x16 blocks)
        mask_large = (partition_map == 0).float().unsqueeze(1)
        
        # mask_inter: 1 where map==1 (16x8 / 8x16 blocks)
        mask_inter = (partition_map == 1).float().unsqueeze(1)
        
        # mask_small: 1 where map==2 (8x8 and smaller)
        mask_small = (partition_map >= 2).float().unsqueeze(1)

        out_large = self.conv_large(x) * mask_large
        out_inter = self.conv_inter(x) * mask_inter
        out_small = self.conv_small(x) * mask_small
        
        return x + out_large + out_inter + out_small

class MVWarp(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_base", None, persistent=False)

    def _grid(self, H, W, device):
        if self._base is None or self._base.shape[1] != H or self._base.shape[2] != W:
            xs = torch.linspace(-1, 1, W, device=device)
            ys = torch.linspace(-1, 1, H, device=device)
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            self._base = torch.stack([xx, yy], dim=-1).unsqueeze(0)
        return self._base

    def forward(self, x, flow):
        # flow is in pixels (B, 2, H, W)
        B, C, H, W = x.shape
        grid = self._grid(H, W, x.device)
        
        vgrid = grid + torch.stack([
            flow[:, 0] * (2.0 / (W - 1)), 
            flow[:, 1] * (2.0 / (H - 1))
        ], dim=-1)
        
        return F.grid_sample(x, vgrid, mode='bilinear', padding_mode='border', align_corners=True)

class CodecConditioner(nn.Module):
    """
    Compresses Codec Priors (Residual + MV Magnitude) into a 
    conditioning feature map for SFT.
    """
    def __init__(self, out_ch=64):
        super().__init__()
        # Input: Residual (1ch) + MV Magnitude (1ch) = 2ch
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(32, out_ch, 3, 1, 1) 
        )

    def forward(self, res, mv):
        # mv: (B, 2, H, W) -> Magnitude: (B, 1, H, W)
        mv_mag = torch.norm(mv, dim=1, keepdim=True)
        
        if res.dim() == 3:
            res = res.unsqueeze(1)
            
        cond_input = torch.cat([res, mv_mag], dim=1)
        return self.net(cond_input)
class ResidualFlowHead(nn.Module):
    """
    Predicts flow purely from features (Since MVs were shown to be useless here).
    """
    def __init__(self, c_feat=64, hidden=64):
        super().__init__()
        # Input: Feat_t-1 (64) + Feat_t (64) = 128 channels
        self.net = nn.Sequential(
            nn.Conv2d(c_feat * 2, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, 3, padding=1) 
        )
        
        # Initialize output to small random values or zero
        # (Zero is safe, lets it learn from scratch)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat_tm1, feat_t):
        # feat: (B, C, H, W)
        x = torch.cat([feat_tm1, feat_t], dim=1)
        flow = self.net(x)
        return flow

class MVSR(nn.Module):
    def __init__(self, mid=64, blocks=15, scale=4):
        super().__init__()
        self.mid = mid
        self.blocks = blocks
        self.scale = scale

        # self.conditioner = CodecConditioner(out_ch=mid)

        # self.conv_first = nn.Conv2d(3, mid, 3, 1, 1)
        # self.relu = nn.ReLU(inplace=True)
        
        # self.sft_blocks = nn.ModuleList([SFTResidualBlock(mid) for _ in range(blocks)])

        self.mvwarp = MVWarp(align_corners=True)
        
        # self.flow_head_fwd = ResidualFlowHead(c_feat=mid, hidden=64)
        # self.flow_head_bwd = ResidualFlowHead(c_feat=mid, hidden=64)
        
        self.feat_extract = nn.Sequential(
            nn.Conv2d(3, mid, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(mid, mid, 3, 1, 1),
            nn.LeakyReLU(0.1, True)
        )
        
        self.mv_refiner = MVRefiner()
        
        self.backward_resblocks = nn.Sequential(*[ResidualBlock(mid) for _ in range(blocks)])
        self.forward_resblocks = nn.Sequential(*[ResidualBlock(mid) for _ in range(blocks)])
        
        # self.prop_fwd = nn.Sequential(
        #     nn.Conv2d(mid * 2 + 1, mid, 3, 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(mid, mid, 3, 1, 1)
        # )
        # self.prop_bwd = nn.Sequential(
        #     nn.Conv2d(mid * 2 + 1, mid, 3, 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(mid, mid, 3, 1, 1)
        # )
        
        self.partion_map = PartitionMap(mid)
        
        self.fusion = nn.Conv2d(mid * 2, mid, 1, 1)
        
        self.up = nn.Sequential(
            nn.Conv2d(mid, mid * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(mid, mid * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(mid, 3, 3, 1, 1)
        )

    def compute_flow(self, mv):
        return self.mv_refiner(mv / 4.0)

    # def extract_features(self, lr_imgs, conds):
    #     # lr_imgs: (B, T, 3, H, W)
    #     # conds:   (B, T, mid, H, W)
    #     B, T, _, H, W = lr_imgs.shape
        
    #     x = lr_imgs.view(B*T, 3, H, W)
    #     c = conds.view(B*T, self.mid, H, W)
        
    #     feat = self.relu(self.conv_first(x))
        
    #     for block in self.sft_blocks:
    #         feat = block(feat, c)
            
    #     return feat.view(B, T, self.mid, H, W)

    def forward(self, imgs, mv_fwd, mv_bwd, residual, partition_maps):
        B, T, C, H, W = imgs.shape
        
        # Extract Features
        feats = self.feat_extract(imgs.view(-1, C, H, W)).view(B, T, -1, H, W)
        
        bwd_features = []
        h_bwd = torch.zeros_like(feats[:, 0])
        
        for t in range(T - 1, -1, -1):
            flow = self.compute_flow(mv_bwd[:, t]) 
            h_bwd = self.mvwarp(h_bwd, flow)
            
            h_bwd = h_bwd + feats[:, t]
            h_bwd = self.backward_resblocks(h_bwd)
            bwd_features.append(h_bwd)
        bwd_features = bwd_features[::-1]

        fwd_features = []
        h_fwd = torch.zeros_like(feats[:, 0])
        
        for t in range(T):
            flow = self.compute_flow(mv_fwd[:, t])
            h_fwd = self.mvwarp(h_fwd, flow)
            
            h_fwd = h_fwd + feats[:, t]
            h_fwd = self.forward_resblocks(h_fwd)
            fwd_features.append(h_fwd)

        outs = []
        for t in range(T):
            fused = self.fusion(torch.cat([fwd_features[t], bwd_features[t]], dim=1))
            
            refined = self.partion_map(fused, partition_maps[:, t])
            
            out = self.up(refined)
            outs.append(out)

        return torch.stack(outs, dim=1)

# ------------------------ Train / Val ------------------------

@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=torch.float16):
    model.eval()
    psnr_sum, ncount = 0.0, 0
    for imgs, gts, mv_fwd, mv_bwd, residual, partition_maps, _, _ in tqdm(loader, desc="val", leave=False):
        imgs     = imgs.to(device, non_blocking=True)       # (B,T,3,H,W)
        gts      = gts.to(device, non_blocking=True)        # (B,T,3,4H,4W)
        mv_fwd   = mv_fwd.to(device, non_blocking=True)     # (B,T,2,H,W)
        mv_bwd   = mv_bwd.to(device, non_blocking=True)     # (B,T,2,H,W)
        residual = residual.to(device, non_blocking=True)   # (B,T,1,H,W)
        partition_maps = partition_maps.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            sr = model(imgs, mv_fwd, mv_bwd, residual, partition_maps)

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
    
    ap.add_argument("--reds_root", type=str, default=None)
    ap.add_argument("--val_reds_root", type=str, default=None)
    ap.add_argument("--flows_root", type=str, default=None)
    ap.add_argument("--val_flows_root", type=str, default=None)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=14)
    ap.add_argument("--crop_lr", type=int, default=96)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda_l1", type=float, default=0.05,
                        help="Weight for the L1 sparsity loss on residual flow")
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
        persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(1, min(2, args.num_workers)), pin_memory=True,
        persistent_workers=True, prefetch_factor=2
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
        for imgs, gts, mv_fwd, mv_bwd, residual, partition_maps, _, _ in pbar:
            imgs           = imgs.to(device, non_blocking=True)
            gts            = gts.to(device, non_blocking=True)
            mv_fwd         = mv_fwd.to(device, non_blocking=True)
            mv_bwd         = mv_bwd.to(device, non_blocking=True)
            residual       = residual.to(device, non_blocking=True)
            partition_maps = partition_maps.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                sr = model(imgs, mv_fwd, mv_bwd, residual, partition_maps)
                loss = charbonnier_loss(sr, gts)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix(
                loss=f"{float(loss.item()):.4f}",
            )
            
            writer.add_scalar("train/loss_total", loss.item(), tb_step)
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
