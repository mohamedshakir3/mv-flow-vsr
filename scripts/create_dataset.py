import csv
import argparse
import subprocess
from pathlib import Path
import numpy as np
import cv2
import av


def run_ffmpeg_encode(hr_dir: Path, out_video: Path,
                      scale_factor: int = 4, crf: int = 23, fps: int = 25):
    """
    Encode HR frames in hr_dir as a single H.264 LR video.
    """
    out_video.parent.mkdir(parents=True, exist_ok=True)
    input_pattern = str(hr_dir / "%08d.png")

    scale_expr = f"scale=iw/{scale_factor}:ih/{scale_factor}"

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-vf", scale_expr,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(out_video),
    ]
    print("Encoding:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_mvs_csv(csv_path: Path, height: int, width: int):
    """
    Parse the CSV produced by your C extract_mvs binary.
    """
    per_frame_fwd = {}
    per_frame_bwd = {}

    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}

        for row in reader:
            if not row or len(row) < len(idx):
                continue

            framenum = int(row[idx["framenum"]])
            frame_idx = framenum - 1

            source = int(row[idx["source"]])
            blockw = int(row[idx["blockw"]])
            blockh = int(row[idx["blockh"]])
            dstx = int(row[idx["dstx"]])
            dsty = int(row[idx["dsty"]])
            motion_x = int(row[idx["motion_x"]])
            motion_y = int(row[idx["motion_y"]])
            motion_scale = int(row[idx["motion_scale"]])

            if motion_scale == 0:
                continue

            dx = motion_x / float(motion_scale)
            dy = motion_y / float(motion_scale)

            if source <= 0:
                frame_flows = per_frame_fwd
            else:
                frame_flows = per_frame_bwd

            if frame_idx not in frame_flows:
                frame_flows[frame_idx] = np.zeros((2, height, width), dtype=np.float32)

            flow = frame_flows[frame_idx]

            x0, y0 = max(0, dstx), max(0, dsty)
            x1, y1 = min(width, dstx + blockw), min(height, dsty + blockh)

            if x0 >= x1 or y0 >= y1:
                continue

            flow[0, y0:y1, x0:x1] = dx
            flow[1, y0:y1, x0:x1] = dy

    return per_frame_fwd, per_frame_bwd


def extract_codec_features(
    video_path: Path,
    mv_extractor: Path,
    out_lr_dir: Path,
    out_mv_fwd_dir: Path,
    out_mv_bwd_dir: Path,
    out_res_dir: Path,
    out_meta_dir: Path,
):
    """
    Decode with PyAV to get LR frames, Residual maps, and Frame Types.
    """
    out_lr_dir.mkdir(parents=True, exist_ok=True)
    out_mv_fwd_dir.mkdir(parents=True, exist_ok=True)
    out_mv_bwd_dir.mkdir(parents=True, exist_ok=True)
    out_res_dir.mkdir(parents=True, exist_ok=True)
    out_meta_dir.mkdir(parents=True, exist_ok=True)

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    width = stream.codec_context.width
    height = stream.codec_context.height

    n_frames = 0
    for _ in container.decode(video=0):
        n_frames += 1
    container.close()

    print(f"Video {video_path.name}: {width}x{height}, {n_frames} frames")

    csv_path = video_path.with_suffix(".mvs.csv")
    cmd = [str(mv_extractor), str(video_path)]
    print("Running MV extractor:", " ".join(cmd))
    with csv_path.open("w") as f:
        subprocess.run(cmd, stdout=f, check=True)

    per_frame_fwd, per_frame_bwd = parse_mvs_csv(csv_path, height, width)

    container = av.open(str(video_path))
    prev_frame_rgb = None
    frame_types = []

    for i, frame in enumerate(container.decode(video=0)):
        img = frame.to_ndarray(format="rgb24")
        h, w, _ = img.shape
        frame_types.append(frame.pict_type)
        
        lr_path = out_lr_dir / f"{i:08d}.png"
        cv2.imwrite(str(lr_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        flow_fwd = per_frame_fwd.get(i, np.zeros((2, h, w), dtype=np.float32))
        flow_bwd = per_frame_bwd.get(i, np.zeros((2, h, w), dtype=np.float32))

        if i > 0:
            np.savez_compressed(out_mv_fwd_dir / f"{i:08d}_mv_fwd.npz", flow_fwd=flow_fwd)
            np.savez_compressed(out_mv_bwd_dir / f"{i:08d}_mv_bwd.npz", flow_bwd=flow_bwd)

        if i > 0 and prev_frame_rgb is not None:
            flow_x = flow_fwd[0]
            flow_y = flow_fwd[1]

            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (grid_x + flow_x).astype(np.float32)
            map_y = (grid_y + flow_y).astype(np.float32)

            prev_bgr = cv2.cvtColor(prev_frame_rgb, cv2.COLOR_RGB2BGR)
            warped_prev = cv2.remap(
                prev_bgr, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            warped_prev_rgb = cv2.cvtColor(warped_prev, cv2.COLOR_BGR2RGB)

            residual = np.mean(
                np.abs(img.astype(np.float32) - warped_prev_rgb.astype(np.float32)),
                axis=2,
            )
            np.save(out_res_dir / f"{i:08d}_res.npy", residual.astype(np.float32))

        prev_frame_rgb = img.copy()

    container.close()
    
    np.save(out_meta_dir / "frame_types.npy", np.array(frame_types, dtype=np.int64))
    print(f"Saved LR, MVs, residuals, and frame_types -> {out_lr_dir.parent}")


def process_reds_train_sharp(
    train_sharp_dir: Path,
    out_root: Path,
    mv_extractor: Path,
    scale_factor: int = 4,
    crf: int = 23,
    fps: int = 25,
):
    seq_dirs = sorted([p for p in train_sharp_dir.iterdir() if p.is_dir()])
    print(seq_dirs)
    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        print(f"\n=== Processing sequence {seq_name} ===")

        video_dir = out_root / "videos"
        video_path = video_dir / f"{seq_name}.mp4"

        run_ffmpeg_encode(seq_dir, video_path, scale_factor=scale_factor, crf=crf, fps=fps)

        seq_out_root = out_root / seq_name
        lr_dir = seq_out_root / "lr"
        mv_fwd_dir = seq_out_root / "mv_fwd"
        mv_bwd_dir = seq_out_root / "mv_bwd"
        res_dir = seq_out_root / "residual"
        meta_dir = seq_out_root / "meta"

        extract_codec_features(
            video_path=video_path,
            mv_extractor=mv_extractor,
            out_lr_dir=lr_dir,
            out_mv_fwd_dir=mv_fwd_dir,
            out_mv_bwd_dir=mv_bwd_dir,
            out_res_dir=res_dir,
            out_meta_dir=meta_dir,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reds_root", type=str, required=True,
                    help="Path to REDS root (containing train_sharp/ or val_sharp/)")
    ap.add_argument("--out_root", type=str, required=True,
                    help="Where to write LR frames + features")
    ap.add_argument("--mv_extractor", type=str, default="./extract_mvs",
                    help="Path to compiled C motion vector extractor binary")
    ap.add_argument("--scale", type=int, default=4,
                    help="Downscale factor (HR -> HR/scale)")
    ap.add_argument("--crf", type=int, default=23,
                    help="H.264 CRF value")
    ap.add_argument("--fps", type=int, default=25,
                    help="Frame rate for encoding")
    args = ap.parse_args()

    reds_root = Path(args.reds_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    mv_extractor = Path(args.mv_extractor)

    train_dir = reds_root / "val_sharp"
    process_reds_train_sharp(
        train_sharp_dir=train_dir,
        out_root=out_root,
        mv_extractor=mv_extractor,
        scale_factor=args.scale,
        crf=args.crf,
        fps=args.fps,
    )

if __name__ == "__main__":
    main()