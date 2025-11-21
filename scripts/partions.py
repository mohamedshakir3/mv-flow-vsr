import os
import numpy as np
import pandas as pd
import av
import argparse
from pathlib import Path
from tqdm import tqdm

def create_partition_maps_from_csv(video_path, csv_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    height = stream.height
    width = stream.width
    num_frames = stream.frames
    container.close()

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    print(f"Processing {video_path.name} ({num_frames} frames)...")
    for i in tqdm(range(num_frames), desc="Generating Maps"):
        
        print(df)
        frame_mvs = df[df['framenum'] == (i + 1)]
        
        h4, w4 = height // 4, width // 4
        frame_map = np.full((h4, w4), 2, dtype=np.uint8)

        if len(frame_mvs) > 0:
            for _, row in frame_mvs.iterrows():
                w_blk = int(row['blockw'])
                h_blk = int(row['blockh'])
                dst_x = int(row['dstx'])
                dst_y = int(row['dsty'])

                area = w_blk * h_blk
                if area >= 256:   p_class = 0
                elif area >= 128: p_class = 1
                else:             p_class = 2

                x_start = dst_x // 4
                y_start = dst_y // 4
                x_end = (dst_x + w_blk) // 4
                y_end = (dst_y + h_blk) // 4

                x_start = max(0, min(x_start, w4))
                y_start = max(0, min(y_start, h4))
                x_end   = max(0, min(x_end, w4))
                y_end   = max(0, min(y_end, h4))

                frame_map[y_start:y_end, x_start:x_end] = p_class

        h16, w16 = height // 16, width // 16
        frame_map_trimmed = frame_map[:h16*4, :w16*4]
        
        final_map = frame_map_trimmed.reshape(h16, 4, w16, 4).max(axis=(1, 3))

        save_path = out_dir / f"{i:08d}.npy"
        np.save(save_path, final_map)

def main():
    base_path = Path("/Users/mohamed/Projects/caper-vsr/outputs/dataset/val")
    video_path = base_path / "videos/000.mp4"
    csv_path   = base_path / "videos/000.mvs.csv"
    
    out_dir    = base_path / "000/partition_maps"

    create_partition_maps_from_csv(video_path, csv_path, out_dir)

if __name__ == "__main__":
    main()