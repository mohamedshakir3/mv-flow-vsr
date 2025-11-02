import os, glob, argparse, subprocess
import numpy as np
import cv2
import torch
from tqdm import tqdm
from train_mv_vsr import MVSR

def imread_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path, img_rgb):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def list_images(dir_path, tmpl):
    return sorted([os.path.join(dir_path, p) for p in os.listdir(dir_path)])


def load_flow_npz(path, H, W):
    if not os.path.exists(path):
        # no flow available: return zeros
        return np.zeros((2, H, W), np.float32)
    z = np.load(path)
    f = z["flow_bwd"].astype(np.float32)  # H,W,2 (dx,dy) in LR pixels
    f = np.transpose(f, (2,0,1))          # 2,H,W
    return f


def run_sequence(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = MVBasicVSR(mid=args.mid, blocks=args.blocks, scale=args.scale).to(device)
    ckpt = torch.load(args.model, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        model = model.to(memory_format=torch.channels_last)

    # list LR frames
    lr_frames = list_images(args.lr_dir, args.img_tmpl)
    lr_frames = [p for p in lr_frames if os.path.exists(p)]
    if len(lr_frames) == 0:
        raise SystemExit(f"No LR frames found in {args.lr_dir}")

    # load one to get size
    H, W = imread_rgb(lr_frames[0]).shape[:2]

    # prepare output dir
    os.makedirs(args.out_dir, exist_ok=True)

    # Autocast dtype
    amp_dtype = torch.bfloat16 if (device.type=='cuda' and args.amp=='bf16') else (
                torch.float16  if (device.type=='cuda' and args.amp=='fp16') else None)

    # streaming TBPTT-style inference to keep memory flat
    K = args.tbptt_k if args.tbptt_k > 0 else len(lr_frames)

    state = None
    t0 = 0
    with torch.inference_mode():
        while t0 < len(lr_frames):
            t1 = min(t0 + K, len(lr_frames))
            for t in range(t0, t1):
                img = imread_rgb(lr_frames[t])
                img_t = torch.from_numpy(img).permute(2,0,1).float()/255.0
                if device.type=='cuda':
                    img_t = img_t.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                else:
                    img_t = img_t.to(device)

                # flow for t>0
                if t == 0:
                    flow_t = None
                else:
                    flow_path = os.path.join(args.flows_dir, args.flow_tmpl.format(t=t))
                    f = load_flow_npz(flow_path, H, W)
                    flow_t = torch.from_numpy(f).to(device)

                # step
                if amp_dtype is not None:
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        sr_t, state = model.step(img_t.unsqueeze(0), flow_t.unsqueeze(0) if flow_t is not None else None, state)
                else:
                    sr_t, state = model.step(img_t.unsqueeze(0), flow_t.unsqueeze(0) if flow_t is not None else None, state)

                # save BGR PNG
                sr = sr_t.squeeze(0).clamp(0,1).mul(255.0).byte().cpu().permute(1,2,0).numpy()
                out_path = os.path.join(args.out_dir, os.path.basename(lr_frames[t]))
                save_rgb(out_path, sr)

            # detach state between segments to bound history (optional)
            if state is not None:
                state = state.detach()
            t0 = t1

    print(f"Wrote {len(lr_frames)} frames to {args.out_dir}")

    if args.video:
        pattern = os.path.join(args.out_dir, os.path.basename(args.img_tmpl).replace('%08d','%08d'))
        cmd = [
            'ffmpeg','-y','-framerate',str(args.fps),
            '-i', os.path.join(args.out_dir, '%08d.png'),
            '-c:v','libx264','-pix_fmt','yuv420p',
            os.path.join(args.out_dir, args.video)
        ]
        try:
            subprocess.run(cmd, check=True)
            print('Wrote video:', os.path.join(args.out_dir, args.video))
        except Exception as e:
            print('ffmpeg failed (is it installed?):', e)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='path to best.pth or epoch_X.pth (state_dict)')
    ap.add_argument('--lr_dir', required=True, help='folder of LR frames (one clip)')
    ap.add_argument('--flows_dir', required=True, help='folder of flow npz for that clip')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--scale', type=int, default=4)
    ap.add_argument('--img_tmpl', default='%08d.png')
    ap.add_argument('--flow_tmpl', default='{t:06d}_mv.npz')
    ap.add_argument('--tbptt_k', type=int, default=0, help='0=full clip, else segment length')
    ap.add_argument('--amp', default='bf16', choices=['bf16','fp16','fp32'])
    ap.add_argument('--mid', type=int, default=64)
    ap.add_argument('--blocks', type=int, default=15)
    ap.add_argument('--fps', type=int, default=25)
    ap.add_argument('--video', default='', help='output mp4 filename inside out_dir (optional)')
    args = ap.parse_args()
    run_sequence(args)
