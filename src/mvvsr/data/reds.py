# encode_pyav.py
import os, cv2, av
import numpy as np
import subprocess
from pathlib import Path
from PIL import Image
import csv


def encode_folder_to_h264(input_folder: str, out_path: str, fps: int = 24, crf: int = 26):
    names = sorted([n for n in os.listdir(input_folder)
                    if n.lower().endswith((".png", ".jpg", ".jpeg"))])
    assert names, f"No images in {input_folder}"
    first = cv2.imread(os.path.join(input_folder, names[0]))
    h, w = first.shape[:2]

    out = av.open(out_path, mode="w")
    stream = out.add_stream("libx264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.codec_context.gop_size = 999
    stream.codec_context.max_b_frames = 0
    stream.codec_context.options = {
        "crf": str(crf),
        "preset": "slow",
        "sc_threshold": "0"                 # avoid surprise I-frames
    }

    for n in names:
        img = cv2.imread(os.path.join(input_folder, n))
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        frame = av.VideoFrame.from_ndarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), format="rgb24")
        for pkt in stream.encode(frame):
            out.mux(pkt)

    for pkt in stream.encode(None):
        out.mux(pkt)
    out.close()

def extract_mvs(clip: str, output: str):
    cmd = ["./extract_mvs", clip]
    with open(output, "w") as f:
        subprocess.run(cmd, stdout=f, text=True, check=True)

def csv_to_npz_flows(csv_path: str, lr_frames_dir: str, out_dir: str):
    """
    Reads CSV rows like:
      framenum,source,blockw,blockh,srcx,srcy,dstx,dsty,flags,motion_x,motion_y,motion_scale
    Keeps only backward refs (source in {-1,0}), converts to pixel flow (dx,dy),
    rasterizes to a dense field per frame, and writes:
      out_dir/{t:06d}_mv.npz  with key 'flow_bwd' -> (H,W,2) float32
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # Infer LR H,W and frame count from the images
    lr_imgs = sorted([p for p in os.listdir(lr_frames_dir)
                      if p.lower().endswith((".png",".jpg",".jpeg"))])
    if not lr_imgs:
        raise RuntimeError(f"No LR frames found in {lr_frames_dir}")
    H, W = Image.open(os.path.join(lr_frames_dir, lr_imgs[0])).size[::-1]
    T = len(lr_imgs)

    # Read CSV and bucket blocks per frame index
    by_frame = {}
    min_fr = None
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                src = int(row["source"])
            except KeyError:
                # Some builds use 'source' or 'source' missing; fallback to -1 (backward)
                src = -1
            if src not in (-1, 0):  # keep backward list only
                continue
            fr = int(row["framenum"])
            bw, bh = int(row["blockw"]), int(row["blockh"])
            x0, y0 = int(row["dstx"]), int(row["dsty"])
            # fixed-point -> pixels
            mscale = float(row.get("motion_scale", 1) or 1)
            dx = int(row["motion_x"]) / mscale
            dy = int(row["motion_y"]) / mscale
            if fr not in by_frame:
                by_frame[fr] = []
            by_frame[fr].append((x0, y0, bw, bh, dx, dy))
            if min_fr is None or fr < min_fr:
                min_fr = fr

    # Align CSV frame numbering to 0..T-1 (CSV may start at 1)
    offset = int(min_fr) if min_fr is not None else 0

    # Build/save dense flow per LR frame index t (0..T-1)
    for t in range(T):
        flow = np.zeros((H, W, 2), np.float32)
        cover = np.zeros((H, W), np.int32)
        blocks = by_frame.get(t + offset, [])
        for x0, y0, bw, bh, dx, dy in blocks:
            x0c = max(0, x0); y0c = max(0, y0)
            x1 = min(W, x0c + bw); y1 = min(H, y0c + bh)
            if x0c >= x1 or y0c >= y1:
                continue
            flow[y0c:y1, x0c:x1, 0] += dx
            flow[y0c:y1, x0c:x1, 1] += dy
            cover[y0c:y1, x0c:x1] += 1

        m = cover > 0
        if m.any():
            flow[m, 0] /= cover[m]
            flow[m, 1] /= cover[m]

        np.savez_compressed(out / f"{t:06d}_mv.npz", flow_bwd=flow)

if __name__ == "__main__":
    training_set_path = "/Volumes/workspace/mohamed/data/REDS/val/val_sharp_bicubic/X4/"
    encodings_path = "./val_encodings/"
    mv_csv_path    = "./val_MVs/"
    flows_root     = "./val_flows/"

    Path(encodings_path).mkdir(parents=True, exist_ok=True)
    Path(mv_csv_path).mkdir(parents=True, exist_ok=True)
    Path(flows_root).mkdir(parents=True, exist_ok=True)

    for frame_dir in sorted(os.listdir(training_set_path)):
        lr_seq_dir   = os.path.join(training_set_path, frame_dir)
        if not os.path.isdir(lr_seq_dir):
            continue

        encoded_clip = os.path.join(encodings_path, f"{frame_dir}.mp4")
        mv_csv       = os.path.join(mv_csv_path, f"{frame_dir}.csv")
        flows_dir    = os.path.join(flows_root, frame_dir)

        print(f"[{frame_dir}] encoding LR -> {encoded_clip}")
        encode_folder_to_h264(lr_seq_dir, encoded_clip)

        print(f"[{frame_dir}] extracting MVs -> {mv_csv}")
        extract_mvs(encoded_clip, mv_csv)

        print(f"[{frame_dir}] rasterizing CSV -> flows (*.npz) in {flows_dir}")
        csv_to_npz_flows(mv_csv, lr_seq_dir, flows_dir)
# if __name__ == "__main__":
#     training_set_path = "/Volumes/workspace/mohamed/data/REDS/train/train_sharp_bicubic/X4/"
#     encodings_path = "./encodings/"
#     mv_path = "./MVs/"
#     for frame_dir in sorted(os.listdir(training_set_path)):
#         encoded_clip = encodings_path + frame_dir + ".mp4"
#         input_path = training_set_path + frame_dir
#         encode_folder_to_h264(input_path, encoded_clip)
#         motion_vectors = mv_path + frame_dir + ".csv"
#         extract_mvs(encoded_clip, motion_vectors)
#     # for clip in sorted(os.listdir("./encodings")):
#     #     encoded_clip = encodings_path + clip
#     #     motion_vectors = mv_path + clip.split(".")[0] + ".csv"
#     #     extract_mvs(encoded_clip, motion_vectors)
        
        