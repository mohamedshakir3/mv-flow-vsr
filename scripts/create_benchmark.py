import os
import glob
import subprocess
import argparse
import numpy as np
import pandas as pd
import cv2
import av
from pathlib import Path
from tqdm import tqdm
import shutil

CLIPS = ["clip_000", "clip_011", "clip_015", "clip_020"]

def encode_reds4_clip(hr_dir, out_mp4_path, scale=4, crf=25):
    """
    Downscales HR images and encodes to H.264 MP4 to generate codec priors.
    """
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    
    images = sorted(list(hr_dir.glob("*.png")))
    if not images:
        raise FileNotFoundError(f"No images found in {hr_dir}")
    input_pattern = str(hr_dir / "%08d.png") 
    
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "25",
        "-i", input_pattern,
        "-vf", f"scale=iw/{scale}:ih/{scale}",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(out_mp4_path)
    ]
    
    if not os.path.exists(str(images[0])):
        print(f"  Renaming {hr_dir} images to 0000.png sequence for FFmpeg...")
        temp_dir = hr_dir.parent / f"{hr_dir.name}_temp"
        temp_dir.mkdir(exist_ok=True)
        for i, img in enumerate(images):
            shutil.copy(img, temp_dir / f"{i:04d}.png")
        input_pattern = str(temp_dir / "%04d.png")
        cmd[4] = input_pattern

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print(f"  Encoded {out_mp4_path.name} (CRF {crf})")

def parse_mvs_into_flows(csv_path, height, width):
    per_frame_fwd = {}
    per_frame_bwd = {}
    df = pd.read_csv(csv_path); df.columns = df.columns.str.strip()

    for _, row in df.iterrows():
        idx = int(row['framenum']) - 1
        scale = float(row['motion_scale'])
        if scale == 0: continue
        dx = row['motion_x'] / scale
        dy = row['motion_y'] / scale
        source = row['source']
        target = per_frame_fwd if source <= 0 else per_frame_bwd
        
        if idx not in target: target[idx] = np.zeros((2, height, width), dtype=np.float32)
        
        dstx, dsty, bw, bh = int(row['dstx']), int(row['dsty']), int(row['blockw']), int(row['blockh'])
        x0, y0 = max(0, dstx), max(0, dsty)
        x1, y1 = min(width, dstx + bw), min(height, dsty + bh)
        
        if x0 < x1 and y0 < y1:
            target[idx][:, y0:y1, x0:x1] = np.array([dx, dy])[:, None, None]

    return per_frame_fwd, per_frame_bwd

def generate_partition_maps(csv_path, num_frames, height, width, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path); df.columns = df.columns.str.strip()

    for i in range(num_frames):
        h4, w4 = height // 4, width // 4
        frame_map = np.full((h4, w4), 2, dtype=np.uint8)
        
        frame_mvs = df[df['framenum'] == (i + 1)]
        if len(frame_mvs) > 0:
            for _, row in frame_mvs.iterrows():
                area = row['blockw'] * row['blockh']
                p_class = 0 if area >= 256 else (1 if area >= 128 else 2)
                
                dstx, dsty, bw, bh = int(row['dstx']), int(row['dsty']), int(row['blockw']), int(row['blockh'])
                x0, y0 = max(0, min(dstx // 4, w4)), max(0, min(dsty // 4, h4))
                x1, y1 = max(0, min((dstx + bw) // 4, w4)), max(0, min((dsty + bh) // 4, h4))
                frame_map[y0:y1, x0:x1] = p_class

        h16, w16 = height // 16, width // 16
        final_map = frame_map[:h16*4, :w16*4].reshape(h16, 4, w16, 4).max(axis=(1, 3))
        np.save(out_dir / f"{i:08d}.npy", final_map)

def process_reds(hr_root, out_root, mv_extractor, crf):
    hr_path = Path(hr_root)
    out_path = Path(out_root)
    mv_bin = Path(mv_extractor).resolve()

    if not mv_bin.exists():
        print(f"Error: {mv_bin} not found.")
        return

    print(f"--- Processing REDS (CRF {crf}) ---")
    
    for clip in CLIPS:
        print(f"\n> Clip: {clip}")
        clip_hr_dir = hr_path / clip
        if not clip_hr_dir.exists():
            print(f"  Skipping (Not found at {clip_hr_dir})")
            continue

        clip_out_dir = out_path / f"REDS4_CRF{crf}" / clip
        dirs = {k: clip_out_dir / k for k in ["lr", "mv_fwd", "mv_bwd", "residual", "partition_maps", "meta"]}
        for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)

        temp_mp4 = clip_out_dir / f"{clip}.mp4"
        encode_reds4_clip(clip_hr_dir, temp_mp4, scale=4, crf=crf)

        csv_path = clip_out_dir / f"{clip}.mvs.csv"
        with open(csv_path, "w") as f:
            subprocess.run([str(mv_bin), str(temp_mp4)], stdout=f, check=True)

        container = av.open(str(temp_mp4))
        stream = container.streams.video[0]
        H, W = stream.codec_context.height, stream.codec_context.width
        num_frames = stream.frames
        container.close()

        fwd_flows, bwd_flows = parse_mvs_into_flows(csv_path, H, W)
        generate_partition_maps(csv_path, num_frames, H, W, dirs["partition_maps"])

        container = av.open(str(temp_mp4))
        prev_rgb = None
        
        frame_types = []

        for i, frame in enumerate(tqdm(container.decode(video=0), total=num_frames)):
            img_rgb = frame.to_ndarray(format="rgb24")
            
            frame_types.append(frame.pict_type)
            
            cv2.imwrite(str(dirs["lr"] / f"{i:08d}.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            
            f_fwd = fwd_flows.get(i, np.zeros((2, H, W), dtype=np.float32))
            f_bwd = bwd_flows.get(i, np.zeros((2, H, W), dtype=np.float32))
            np.savez_compressed(dirs["mv_fwd"] / f"{i:08d}_mv_fwd.npz", flow_fwd=f_fwd)
            np.savez_compressed(dirs["mv_bwd"] / f"{i:08d}_mv_bwd.npz", flow_bwd=f_bwd)
            
            if i > 0 and prev_rgb is not None:
                grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
                map_x = (grid_x + f_fwd[0]).astype(np.float32)
                map_y = (grid_y + f_fwd[1]).astype(np.float32)
                prev_bgr = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2BGR)
                warped = cv2.remap(prev_bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                diff = np.abs(img_rgb.astype(np.float32) - cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32))
                res = np.mean(diff, axis=2)
                np.save(dirs["residual"] / f"{i:08d}_res.npy", res.astype(np.float32))
            else:
                np.save(dirs["residual"] / f"{i:08d}_res.npy", np.zeros((H, W), dtype=np.float32))
            
            prev_rgb = img_rgb.copy()
        
        container.close()
        
        np.save(dirs["meta"] / "frame_types.npy", np.array(frame_types, dtype=np.int64))
        print(f"  Saved metadata to {dirs['meta']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr_root", required=True, help="Path to REDS GT folder")
    parser.add_argument("--out_root", default="outputs/benchmarks", help="Where to save processed data")
    parser.add_argument("--crf", type=int, default=25, help="Compression level (15, 25, 35)")
    parser.add_argument("--mv_extractor", default="./extract_mvs", help="Path to C binary")
    args = parser.parse_args()

    process_reds(args.hr_root, args.out_root, args.mv_extractor, args.crf)