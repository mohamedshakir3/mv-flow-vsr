import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
import numpy as np
import os

# --- 1. SETUP ---
# Adjust these paths to your project
GT_ROOT = "inputs/REDS4"
LR_ROOT = "outputs/benchmarks/REDS4_CRF18" # Where you saved LR frames
OURS_ROOT = "results/reds4"

def get_crop(img, x, y, size):
    # 'size' determines zoom level. Larger size = "Zoom out" (more context)
    return img.crop((x, y, x+size, y+size))

def plot_multi_crop(clip, frame_idx, crop_list, forced_path=None):
    """
    crop_list: List of dicts [{'x':.., 'y':.., 'size':.., 'color':..}, ...]
    """
    frame_name = f"{frame_idx:08d}.png"
    
    # --- Load Images ---
    path_gt = f"{GT_ROOT}/{clip}/{frame_name}"
    path_lr = f"{LR_ROOT}/{clip}/lr/{frame_name}"
    path_ours = f"{OURS_ROOT}/{clip}/{frame_name}"
    
    if not os.path.exists(path_gt): return print(f"Missing GT: {path_gt}")
    
    img_gt = Image.open(path_gt)
    img_ours = Image.open(path_ours)
    
    # Generate Bicubic Baseline
    img_lr_cv = cv2.imread(path_lr)
    h, w = img_lr_cv.shape[:2]
    img_bic_cv = cv2.resize(img_lr_cv, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
    img_bic = Image.fromarray(cv2.cvtColor(img_bic_cv, cv2.COLOR_BGR2RGB))

    # Define Models to Compare
    # Format: (Label, ImageObject)
    models = [("Bicubic", img_bic), ("Ours", img_ours), ("GT", img_gt)]
    
    # Optional: Add Forced model for City if provided
    if forced_path and os.path.exists(forced_path):
        img_force = Image.open(forced_path)
        # Insert before GT
        models.insert(2, ("Ours (Forced)", img_force))

    # --- Plotting Layout ---
    num_crops = len(crop_list)
    num_models = len(models)
    
    # Grid: Left huge col for Full Image, Right cols for crops
    fig = plt.figure(figsize=(14, 6))
    
    # 1. Plot Full Image (Left)
    # Spans all rows
    ax_full = plt.subplot2grid((num_crops, num_models + 2), (0, 0), rowspan=num_crops, colspan=2)
    ax_full.imshow(img_gt)
    ax_full.set_title(f"{clip.capitalize()} (Frame {frame_idx})", fontsize=14)
    ax_full.axis('off')

    # Draw boxes on Full Image
    for crop in crop_list:
        x, y, s, c = crop['x'], crop['y'], crop['size'], crop['color']
        rect = patches.Rectangle((x, y), s, s, linewidth=3, edgecolor=c, facecolor='none')
        ax_full.add_patch(rect)

    # 2. Plot Crops (Right Grid)
    for row, crop in enumerate(crop_list):
        x, y, size, color = crop['x'], crop['y'], crop['size'], crop['color']
        
        for col, (name, img) in enumerate(models):
            # Extract patch
            patch = get_crop(img, x, y, size)
            
            # Position: Row=row, Col=col+2 (skip the first 2 cols used by full image)
            ax = plt.subplot2grid((num_crops, num_models + 2), (row, col + 2))
            
            ax.imshow(patch)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Colored border to match the box
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)
            
            # Titles only on top row
            if row == 0:
                ax.set_title(name, fontsize=12, fontweight='bold')

    outfile = f"figure_{clip}_zoom.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"Saved {outfile}")

# --- RUN CONFIGURATION ---
if __name__ == "__main__":
    
    plot_multi_crop(
        clip="clip_011", 
        frame_idx=20, 
        crop_list=[
            {'x': 350, 'y': 180, 'size': 120, 'color': 'red'},  
            {'x': 1000,  'y': 120,  'size': 120, 'color': 'cyan'}
        ]
    )

    plot_multi_crop(
        clip="clip_000", 
        frame_idx=10,
        crop_list=[
            {'x': 500, 'y': 400, 'size': 140, 'color': 'red'},
            {'x': 100, 'y': 400, 'size': 140, 'color': 'lime'}
        ]
    )
    plot_multi_crop(
        clip="clip_015", 
        frame_idx=20, 
        crop_list=[
            {'x': 350, 'y': 180, 'size': 120, 'color': 'red'},  
            
            {'x': 675,  'y': 400,  'size': 120, 'color': 'cyan'}
        ]
    )
    plot_multi_crop(
        clip="clip_020", 
        frame_idx=20, 
        crop_list=[
            {'x': 350, 'y': 180, 'size': 120, 'color': 'red'},  
            
            {'x': 675,  'y': 400,  'size': 120, 'color': 'cyan'}
        ]
    )