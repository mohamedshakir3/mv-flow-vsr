# Utilities for using codec motion vectors (from LR encodes).
# - Load per-frame flows saved as npz (key: 'flow_bwd', shape HxWx2 in LR pixels)
# - Resize flows for feature maps (divide magnitudes by scale)
# - Warp tensors with backward flow (t -> t-1) using grid_sample
# - Optional gating to avoid harmful warps on low-motion frames

from __future__ import annotations
import os
from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------
# Flow I/O
# ----------------------------

def _npz_path(root: Union[str, Path], seq: str, t: int) -> Path:
    return Path(root) / seq / f"{t:06d}_mv.npz"

@lru_cache(maxsize=4096)
def load_flow_npz(path: Union[str, Path]) -> torch.Tensor:
    """
    Load a single-frame flow from an .npz written by your CSV->flows step.
    Returns (1, 2, H, W) float32 in *LR pixels*.
    """
    data = np.load(path, allow_pickle=False)
    flow = data["flow_bwd"]  # (H, W, 2), dtype float32
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Bad flow shape in {path}: {flow.shape}")
    # (H,W,2) -> (1,2,H,W)
    t = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).contiguous().float()
    return t

def load_flow_for_index(
    flows_root: Union[str, Path],
    seq: str,
    t: int,
    expected_hw: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Load flow for frame t (backward: t -> t-1). If missing, returns zeros (1,2,H,W).
    expected_hw is required to produce correctly shaped zeros when the file is missing.
    """
    p = _npz_path(flows_root, seq, t)
    if p.exists():
        return load_flow_npz(p)
    if expected_hw is None:
        raise FileNotFoundError(f"Missing flow {p} and expected_hw not provided.")
    H, W = expected_hw
    return torch.zeros(1, 2, H, W, dtype=torch.float32)

def resize_flow_to(
    flow_lr_pix: torch.Tensor,
    size_hw: Tuple[int, int],
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Resize a pixel-space flow to 'size_hw' and divide magnitudes accordingly.
    flow_lr_pix: (B,2,Hlr,Wlr), values in LR pixels.
    size_hw: (Hout, Wout) -> typically feature map resolution.
    """
    assert flow_lr_pix.dim() == 4 and flow_lr_pix.size(1) == 2, f"Expected (B,2,H,W), got {flow_lr_pix.shape}"
    Hlr, Wlr = flow_lr_pix.shape[-2], flow_lr_pix.shape[-1]
    Hout, Wout = int(size_hw[0]), int(size_hw[1])
    # Single uniform scale (assumes isotropic downscale from LR->features)
    scale = Wlr / float(Wout)
    # Resize spatially, then divide by scale to convert pixel magnitudes
    flow_resized = F.interpolate(flow_lr_pix, size=(Hout, Wout), mode="bilinear", align_corners=align_corners)
    return flow_resized / scale

def _base_grid(B: int, H: int, W: int, device: torch.device, dtype: torch.dtype, align_corners: bool) -> torch.Tensor:
    # Create normalized base grid in [-1,1]x[-1,1]
    if align_corners:
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    else:
        xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W * 2 - 1
        ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H * 2 - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)  # (H,W,2)
    return grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B,H,W,2)

def warp_with_flow(
    x: torch.Tensor,
    flow_xy_pix: torch.Tensor,
    align_corners: bool = True,
    padding_mode: str = "border",
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Backward warp: sample x at (grid - flow) where flow is in *pixels*.
    x:             (B,C,H,W)
    flow_xy_pix:   (B,2,H,W) with flow[...,0]=dx (cols +right), flow[...,1]=dy (rows +down)
    returns:       (B,C,H,W) warped x
    """
    assert x.dim() == 4 and flow_xy_pix.dim() == 4, f"Bad shapes x={x.shape} flow={flow_xy_pix.shape}"
    B, C, H, W = x.shape
    assert flow_xy_pix.shape[0] == B and flow_xy_pix.shape[2:] == (H, W), \
        f"Flow batch/size mismatch: {flow_xy_pix.shape} vs {x.shape}"

    # Normalize flow from pixels to [-1,1] grid units
    # Note: when align_corners=True, scale is 2/(W-1) and 2/(H-1)
    epsW = max(W - 1, 1)
    epsH = max(H - 1, 1)
    nx = flow_xy_pix[:, 0, :, :] * (2.0 / epsW)
    ny = flow_xy_pix[:, 1, :, :] * (2.0 / epsH)

    grid = _base_grid(B, H, W, x.device, x.dtype, align_corners)
    grid = torch.stack([grid[..., 0] + nx, grid[..., 1] + ny], dim=-1)  # (B,H,W,2)
    return F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

def warp_with_flow_gated(
    x: torch.Tensor,
    flow_xy_pix: torch.Tensor,
    min_mag: float = 0.25,
    min_frac: float = 0.002,
    align_corners: bool = True,
    padding_mode: str = "border",
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Same as warp_with_flow, but if motion is negligible (few pixels above min_mag),
    returns a blend: gate*warped + (1-gate)*x (per-sample gate).
    """
    B, _, H, W = x.shape
    mag = torch.linalg.vector_norm(flow_xy_pix, dim=1)  # (B,H,W)
    frac = (mag > min_mag).float().mean(dim=(1, 2), keepdim=True)  # (B,1,1)
    gate = (frac >= min_frac).to(x.dtype).view(B, 1, 1, 1)

    warped = warp_with_flow(x, flow_xy_pix, align_corners=align_corners, padding_mode=padding_mode, mode=mode)
    return gate * warped + (1.0 - gate) * x

class MVProvider:
    """
    Simple disk-backed flow provider.
    Expects files at: flows_root/<seq>/<t:06d>_mv.npz  with key 'flow_bwd' (HxWx2) in LR pixels.

    Example:
      provider = MVProvider("./flows")
      flow_t = provider("000", t).to(device)  # (1,2,H,W)
    """
    def __init__(self, flows_root: Union[str, Path]):
        self.flows_root = Path(flows_root)

    def __call__(self, seq: str, t: int, expected_hw: Optional[Tuple[int,int]] = None) -> torch.Tensor:
        return load_flow_for_index(self.flows_root, seq, t, expected_hw)


class MVFlowNet(torch.nn.Module):
    """
    Minimal estimator shim that returns cached MVs instead of learned optical flow.
    Intended for recurrent super res where forward(I_t, I_t-1) -> flow_t.
    """
    def __init__(self):
        super().__init__()
        self.current_flow: Optional[torch.Tensor] = None  # (B,2,H,W) in LR pixels

    def set_flow(self, flow_lr_pix: torch.Tensor):
        # flow should be (B,2,H,W) on the right device/dtype
        self.current_flow = flow_lr_pix

    def forward(self, img_t: torch.Tensor, img_tm1: torch.Tensor, flow_override: Optional[torch.Tensor] = None):
        """
        Return (flow_t), ignoring images. Keep the signature compatible with existing code.
        """
        flow = flow_override if flow_override is not None else self.current_flow
        if flow is None:
            raise RuntimeError("MVFlowNet: no flow set. Call set_flow(...) or pass flow_override.")
        return flow
