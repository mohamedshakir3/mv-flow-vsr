import os
import glob
import argparse
import subprocess
import numpy as np
import cv2
import torch
from tqdm import tqdm
from src.mvvsr.models.mv_vsr import MVSR
import torch.nn.functional as F

def imread_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def save_rgb(path, img_rgb):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, bgr)

def load_mv(path, h, w):
    """Loads (2, H, W) motion vector field."""
    if not os.path.exists(path):
        return np.zeros((2, h, w), dtype=np.float32)
    
    data = np.load(path)
    if "flow_fwd" in data:
        f = data["flow_fwd"]
    elif "flow_bwd" in data:
        f = data["flow_bwd"]
    else:
        return np.zeros((2, h, w), dtype=np.float32)

    if f.shape[0] != 2: 
        f = np.transpose(f, (2, 0, 1))
        
    return f.astype(np.float32)

def load_residual(path, h, w):
    """Loads (1, H, W) residual map."""
    if not os.path.exists(path):
        return np.zeros((1, h, w), dtype=np.float32)
    res = np.load(path).astype(np.float32)
    if res.ndim == 2:
        res = res[np.newaxis, ...] # (1, H, W)
    return res

def load_partition(path, h, w):
    """Loads (H, W) partition map (integers)."""
    if not os.path.exists(path):
        return np.full((h, w), 2, dtype=np.int64)
    
    p = np.load(path).astype(np.int64)
        
    return p

def run_sequence(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model from {args.model}...")
    model = MVSR(mid=args.mid, blocks=args.blocks, scale=args.scale).to(device)
    
    ckpt = torch.load(args.model, map_location='cpu')
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
        
    new_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_dict[name] = v
        
    model.load_state_dict(new_dict, strict=True)
    model.eval()

    clip_root = args.clip_root
    lr_dir = os.path.join(clip_root, "lr")
    
    lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*.png")))
    if len(lr_paths) == 0:
        raise ValueError(f"No images found in {lr_dir}")

    sample = imread_rgb(lr_paths[0])
    H, W, _ = sample.shape
    T = len(lr_paths)

    print(f"Processing sequence T={T} frames, {W}x{H}")

    imgs_t = []
    mv_fwd_t = []
    mv_bwd_t = []
    part_t = []

    for i, path in enumerate(tqdm(lr_paths, desc="Loading Data")):
        img = imread_rgb(path)
        imgs_t.append(torch.from_numpy(img).permute(2,0,1).float()/255.0)
        
        stem = os.path.splitext(os.path.basename(path))[0]
        
        p_fwd = os.path.join(clip_root, "mv_fwd", f"{stem}_mv_fwd.npz")
        p_bwd = os.path.join(clip_root, "mv_bwd", f"{stem}_mv_bwd.npz")
        
        mv_fwd_t.append(torch.from_numpy(load_mv(p_fwd, H, W)))
        mv_bwd_t.append(torch.from_numpy(load_mv(p_bwd, H, W)))

        p_part = os.path.join(clip_root, "partition_maps", f"{stem}.npy")
        if not os.path.exists(p_part):
             p_part = os.path.join(clip_root, "partition_maps", f"{stem}_part.npy")
             
        part_t.append(torch.from_numpy(load_partition(p_part, H, W)))

    imgs = torch.stack(imgs_t, dim=0).unsqueeze(0)         # (1, T, 3, H, W)
    mv_fwd = torch.stack(mv_fwd_t, dim=0).unsqueeze(0)     # (1, T, 2, H, W)
    mv_bwd = torch.stack(mv_bwd_t, dim=0).unsqueeze(0)     # (1, T, 2, H, W)
    part = torch.stack(part_t, dim=0).unsqueeze(0)         # (1, T, H, W)
    
    part = torch.full_like(part, 2)
    mv_fwd = torch.zeros_like(mv_fwd)
    mv_bwd = torch.zeros_like(mv_bwd)
    
    # 0: 22.911
    # 1: 23.857
    # 2: 26.248
    # mix: 25.775
    # mix no MV: 25.713

    imgs = imgs.to(device)
    mv_fwd = mv_fwd.to(device)
    mv_bwd = mv_bwd.to(device)
    part = part.to(device)

    amp_dtype = torch.bfloat16 if (device.type=='cuda' and args.amp=='bf16') else None
    B, T, H, W = part.shape
    with torch.no_grad():
        if amp_dtype is not None:
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                output = model(imgs, mv_fwd, mv_bwd, part)
        else:
            output = model(imgs, mv_fwd, mv_bwd, part)
    # Output shape: (1, T, 3, 4H, 4W)
    output = output.squeeze(0).cpu() # (T, 3, 4H, 4W)
    
    print(f"Saving {T} frames to {args.out_dir}...")
    os.makedirs(args.out_dir, exist_ok=True)
    
    for i in range(T):
        tensor = output[i].clamp(0, 1).numpy() # (3, 4H, 4W)
        tensor = np.transpose(tensor, (1, 2, 0)) # (4H, 4W, 3)
        tensor = (tensor * 255.0).astype(np.uint8)
        
        fn = os.path.basename(lr_paths[i])
        save_path = os.path.join(args.out_dir, fn)
        save_rgb(save_path, tensor)

    if args.video:
        print("Encoding video...")
        cmd = [
            'ffmpeg', '-y', '-framerate', str(args.fps),
            '-i', os.path.join(args.out_dir, '%08d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            os.path.join(args.out_dir, args.video)
        ]
        subprocess.run(cmd, check=False)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='path to best.pth')
    ap.add_argument('--clip_root', required=True, help='Root of clip (containing lr, mv_fwd, etc)')
    ap.add_argument('--out_dir', required=True, help='Where to save pngs')
    ap.add_argument('--scale', type=int, default=4)
    ap.add_argument('--mid', type=int, default=64)
    ap.add_argument('--blocks', type=int, default=15)
    ap.add_argument('--fps', type=int, default=25)
    ap.add_argument('--video', default='out.mp4', help='Name of output video file')
    ap.add_argument('--amp', default='bf16')
    args = ap.parse_args()
    
    run_sequence(args)