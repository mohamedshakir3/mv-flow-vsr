# mv_warp_debug.py
import os, argparse, numpy as np, cv2, torch
import torch.nn.functional as F
from pathlib import Path
from math import log10

def read_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def to_tensor01(img_rgb):  # HWC uint8 -> 1xCxHxW float[0,1]
    t = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).float()/255.0
    return t

def flow_to_hsv_bgr(flow):  # flow: HxWx2 (dx,dy) -> BGR image
    dx = flow[...,0].astype(np.float32)
    dy = flow[...,1].astype(np.float32)
    mag, ang = cv2.cartToPolar(dx, dy, angleInDegrees=True)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[...,0] = (ang / 2).astype(np.uint8)   # 0..179
    hsv[...,1] = 255
    # scale magnitude for visibility; tune factor if needed
    hsv[...,2] = np.clip(mag * 12, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def warp_with_flow(x, flow_xy_pix, align_corners=True):
    # x: (1,C,H,W), flow: (1,2,H,W) in pixels, backward (t -> t-1)
    B,C,H,W = x.shape
    nx = flow_xy_pix[:,0] * (2.0 / max(W-1,1))
    ny = flow_xy_pix[:,1] * (2.0 / max(H-1,1))
    if align_corners:
        xs = torch.linspace(-1,1,W,device=x.device)
        ys = torch.linspace(-1,1,H,device=x.device)
    else:
        xs = (torch.arange(W,device=x.device)+0.5)/W*2-1
        ys = (torch.arange(H,device=x.device)+0.5)/H*2-1
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    base = torch.stack([xx,yy], dim=-1).unsqueeze(0)  # (1,H,W,2)
    grid = torch.stack([base[...,0]+nx, base[...,1]+ny], dim=-1)
    warped = F.grid_sample(
        x, grid, mode='bilinear', padding_mode='border', align_corners=align_corners
    )
    return warped

def smooth_and_clamp_flow(F, sigma=0.8, p=95):
    f = F[0].permute(1,2,0).detach().cpu().numpy()  # H,W,2
    f[...,0] = cv2.GaussianBlur(f[...,0], (5,5), sigma)
    f[...,1] = cv2.GaussianBlur(f[...,1], (5,5), sigma)
    mag = np.linalg.norm(f, axis=-1)
    m95 = np.percentile(mag, p)
    mask = mag > m95
    f[mask] *= (m95 / (mag[mask] + 1e-6))[:, None]
    return torch.from_numpy(f).permute(2,0,1).unsqueeze(0).to(F)

def psnr(x, y, mask=None):
    # x,y: torch (1,3,H,W) in [0,1]; mask optional torch (1,1,H,W) {0,1}
    if mask is not None:
        n = mask.sum().item()
        if n < 1: return float('nan')
        diff2 = ((x - y)**2 * mask).sum() / n
    else:
        diff2 = ((x - y)**2).mean()
    mse = diff2.item()
    if mse <= 1e-12: return 99.0
    return 10 * log10(1.0 / mse)

def main(args):
    t = args.t
    if t == 0:
        raise SystemExit("t=0 is an I-frame (no MVs). Choose t>=1.")

    lr_list = sorted([p for p in os.listdir(args.lr_dir) if p.lower().endswith(('.png','.jpg','.jpeg'))])
    f_tm1 = os.path.join(args.lr_dir, lr_list[t-1])
    f_t   = os.path.join(args.lr_dir, lr_list[t])

    I_tm1 = read_rgb(f_tm1)  # HxWx3
    I_t   = read_rgb(f_t)
    H, W = I_t.shape[:2]

    # Load flow (HxWx2) in LR pixels
    npz_path = os.path.join(args.flows_dir, f"{t:06d}_mv.npz")
    z = np.load(npz_path)
    flow = z["flow_bwd"].astype(np.float32)
    assert flow.shape[:2] == (H, W), f"Flow size {flow.shape[:2]} != frame {H,W}"

    # Tensors
    X_tm1 = to_tensor01(I_tm1).cuda() if args.cuda else to_tensor01(I_tm1)
    X_t   = to_tensor01(I_t).cuda()   if args.cuda else to_tensor01(I_t)
    F_t   = torch.from_numpy(flow).permute(2,0,1).unsqueeze(0).float()
    if args.cuda: F_t = F_t.cuda()

    # Warp prev -> current
    warped = warp_with_flow(X_tm1, F_t)

    # Baselines & masks
    psnr_identity = psnr(X_tm1, X_t)  # no warp
    psnr_warp     = psnr(warped, X_t)

    # Motion mask: |flow| > thresh (in LR pixels)
    thr = args.motion_thresh
    mag = torch.linalg.vector_norm(F_t, dim=1, keepdim=True)  # (1,1,H,W)
    mask = (mag > thr).float()
    psnr_id_m = psnr(X_tm1, X_t, mask)
    psnr_wp_m = psnr(warped, X_t, mask)

    print(f"PSNR identity     : {psnr_identity:.2f} dB")
    print(f"PSNR warped       : {psnr_warp:.2f} dB")
    print(f"PSNR identity (mv): {psnr_id_m:.2f} dB   [mask |flow|>{thr}]")
    print(f"PSNR warped   (mv): {psnr_wp_m:.2f} dB   [mask |flow|>{thr}]")

    warped_np = (warped.clamp(0,1).cpu().numpy()[0].transpose(1,2,0) * 255).astype(np.uint8)
    It_np     = (X_t.clamp(0,1).cpu().numpy()[0].transpose(1,2,0) * 255).astype(np.uint8)

    # Residual heatmap |I_t - warped|
    resid = cv2.absdiff(It_np, warped_np).mean(axis=2).astype(np.uint8)  # grayscale 0..255
    resid_color = cv2.applyColorMap(resid, cv2.COLORMAP_MAGMA)

    # Flow HSV viz
    flow_bgr = flow_to_hsv_bgr(flow)

    # Tile into a single image (BGR for cv2.imwrite)
    top = np.concatenate([cv2.cvtColor(It_np, cv2.COLOR_RGB2BGR),
                          cv2.cvtColor(warped_np, cv2.COLOR_RGB2BGR)], axis=1)
    bottom = np.concatenate([resid_color, flow_bgr], axis=1)
    panel = np.concatenate([top, bottom], axis=0)

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, panel)
    print(f"Saved debug panel → {args.out}")
    print("Top-left: I_t   | Top-right: warp(I_{t-1})")
    print("Bottom-left: |diff| (hot = worse) | Bot-right: flow HSV")

    return (psnr_identity, psnr_warp)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr_dir", required=True, help="Folder with LR frames (e.g., .../train_sharp_bicubic/X4/000)")
    ap.add_argument("--flows_dir", required=True, help="Folder with per-frame flows .npz")
    ap.add_argument("--t", type=int, required=True, help="Target frame index t (use t>=1)")
    ap.add_argument("--out", default="debug_panel.png")
    ap.add_argument("--motion_thresh", type=float, default=0.25, help="pixels; mask for moving areas")
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()
    main(args)
