import os
import matplotlib.pyplot as plt
from mv_warp import main
import csv
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Args:
    lr_dir: str
    flows_dir: str
    t: int
    out: str
    cuda: bool = False
    try_flip: bool = False
    motion_thresh: float = 0.25

if __name__ == "__main__":
    lr_root = "/Volumes/workspace/mohamed/data/REDS/train/train_sharp_bicubic/X4/"
    flows_path = "flows/"
    output_path = "psnr/"

    for clip in sorted(os.listdir(lr_root)):
        flows_dir = os.path.join(flows_path, clip)
        lr_dir = os.path.join(lr_root, clip)
        output = os.path.join(output_path, clip)
        Path(output).mkdir(parents=True, exist_ok=True)

        csv_path = os.path.join(output, "psnr.csv")
        plot_fig = os.path.join(output, "psnr_plot.png")

        psnr = []
        for t in range(1, 100):
            out = os.path.join(output, f"t{t:04d}.png")
            print(out)
            args = Args(lr_dir, flows_dir, t, out)
            psnr_identity, psnr_warp = main(args)
            psnr.append((t, psnr_warp, psnr_identity))

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "psnr_warped", "psnr_identity"])
            writer.writerows(psnr)

        x  = [t for t, _, _ in psnr]
        yW = [w for _, w, _ in psnr]   # warped
        yI = [i for _, _, i in psnr]   # identity

        fig, ax = plt.subplots()

        ax.plot(x, yW, label="PSNR (Warped)", linewidth=1.8, marker="o", markersize=3)
        ax.plot(x, yI, label="PSNR (Identity)", linewidth=1.8, marker="o", markersize=3)
        ax.legend(loc="best", frameon=True)

        ax.set_xlabel("Frame t")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(f"PSNR vs Frame — clip {clip}")
        ax.grid(True, linewidth=0.6, alpha=0.5)

        # keep the y-axis a bit padded
        ymin = min(min(yW), min(yI))
        ymax = max(max(yW), max(yI))
        ax.set_ylim(ymin - 0.5, ymax + 0.5)

        fig.tight_layout()
        plt.savefig(plot_fig, dpi=180)
        plt.close(fig)
