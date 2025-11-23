import os, glob, random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


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