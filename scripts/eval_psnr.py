import os, argparse, glob, math, csv
import numpy as np
import cv2
from tqdm import tqdm

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}

def list_images(d, pattern="*"):
    files = [p for p in glob.glob(os.path.join(d, pattern)) if os.path.splitext(p)[1].lower() in IMG_EXTS]
    files.sort()
    return files

def to_y_channel_rgb(rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return ycrcb[..., 0]

def shave_border(img, shave):
    if shave is None or shave <= 0:
        return img
    h, w = img.shape[:2]
    if h <= 2 * shave or w <= 2 * shave:
        return img
    return img[shave:h - shave, shave:w - shave]

def psnr_from_arrays(a, b, maxv=255.0):
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-10:
        return 99.0
    return 20.0 * math.log10(maxv) - 10.0 * math.log10(mse)

def main():
    ap = argparse.ArgumentParser(description="Evaluate PSNR between SR frames and GT frames.")
    ap.add_argument("--sr_dir", required=True, help="directory of SR frames (e.g., outputs from inference)")
    ap.add_argument("--gt_dir", required=True, help="directory of GT HR frames (e.g., REDS val_sharp/<clip>)")
    ap.add_argument("--mode", default="y", choices=["y", "rgb"], help="PSNR on Y channel (YCrCb) or RGB average")
    ap.add_argument("--scale", type=int, default=4, help="upscale factor (used as default shave)")
    ap.add_argument("--shave", type=int, default=None, help="border shave in pixels (defaults to --scale if not set)")
    ap.add_argument("--csv", default="", help="optional path to write per-frame CSV metrics")
    ap.add_argument("--pattern", default="*.png", help="glob pattern for frames (default: *.png)")
    args = ap.parse_args()

    shave = args.scale if args.shave is None else args.shave

    sr_files = list_images(args.sr_dir, args.pattern)
    gt_files = list_images(args.gt_dir, args.pattern)
    if not sr_files or not gt_files:
        raise SystemExit("No images found in one or both directories.")

    sr_map = {os.path.basename(p): p for p in sr_files}
    gt_map = {os.path.basename(p): p for p in gt_files}
    names = sorted(set(sr_map.keys()) & set(gt_map.keys()))
    if not names:
        raise SystemExit("No overlapping filenames between SR and GT. Make sure names match (e.g., 00000000.png).")

    rows, psnrs = [], []
    for name in tqdm(names, desc="eval"):
        sr_bgr = cv2.imread(sr_map[name], cv2.IMREAD_COLOR)
        gt_bgr = cv2.imread(gt_map[name], cv2.IMREAD_COLOR)
        if sr_bgr is None or gt_bgr is None:
            continue
        sr_rgb = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2RGB)
        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)

        if args.mode == "y":
            sr_y = shave_border(to_y_channel_rgb(sr_rgb), shave)
            gt_y = shave_border(to_y_channel_rgb(gt_rgb), shave)
            ps = psnr_from_arrays(sr_y, gt_y, maxv=255.0)
        else:
            sr_rgb = shave_border(sr_rgb, shave)
            gt_rgb = shave_border(gt_rgb, shave)
            ps = (
                psnr_from_arrays(sr_rgb[..., 0], gt_rgb[..., 0])
                + psnr_from_arrays(sr_rgb[..., 1], gt_rgb[..., 1])
                + psnr_from_arrays(sr_rgb[..., 2], gt_rgb[..., 2])
            ) / 3.0

        psnrs.append(ps)
        rows.append((name, ps))

    mean_psnr = float(np.mean(psnrs)) if psnrs else 0.0
    print(f"Frames evaluated: {len(psnrs)} | Mean PSNR ({args.mode}, shave={shave}): {mean_psnr:.3f} dB")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", f"psnr_{args.mode}"])
            for name, ps in rows:
                w.writerow([name, f"{ps:.6f}"])
            w.writerow(["mean", f"{mean_psnr:.6f}"])
        print("Wrote CSV:", args.csv)

if __name__ == "__main__":
    main()
