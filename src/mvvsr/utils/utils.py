
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1), nn.ReLU(inplace=False),
            nn.Conv2d(nf, nf, 3, 1, 1)
        )
    def forward(self, x): return x + self.body(x)

class MVRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, 1, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, mv):
        noise = self.net(mv)
        return mv + noise
    
class PartitionMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_large = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
        
        self.conv_inter = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
        
        self.conv_small = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )

    def forward(self, x, partition_map):
        """
        x: (B, C, H, W)
        partition_map: (B, H_map, W_map) Integer tensor {0, 1, 2}
        """
        B, C, H, W = x.shape
        
        if partition_map.shape[-1] != W:
            partition_map = F.interpolate(
                partition_map.unsqueeze(1).float(), 
                size=(H, W), 
                mode='nearest'
            ).squeeze(1).long()
            
        mask_large = (partition_map == 0).float().unsqueeze(1)
        
        mask_inter = (partition_map == 1).float().unsqueeze(1)
        
        mask_small = (partition_map >= 2).float().unsqueeze(1)

        out_large = self.conv_large(x) * mask_large
        out_inter = self.conv_inter(x) * mask_inter
        out_small = self.conv_small(x) * mask_small
        
        return x + out_large + out_inter + out_small

class MVWarp(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_base", None, persistent=False)

    def _grid(self, H, W, device):
        if self._base is None or self._base.shape[1] != H or self._base.shape[2] != W:
            xs = torch.linspace(-1, 1, W, device=device)
            ys = torch.linspace(-1, 1, H, device=device)
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            self._base = torch.stack([xx, yy], dim=-1).unsqueeze(0)
        return self._base

    def forward(self, x, flow):
        # flow is in pixels (B, 2, H, W)
        B, C, H, W = x.shape
        grid = self._grid(H, W, x.device)
        
        vgrid = grid + torch.stack([
            flow[:, 0] * (2.0 / (W - 1)), 
            flow[:, 1] * (2.0 / (H - 1))
        ], dim=-1)
        
        return F.grid_sample(x, vgrid, mode='bilinear', padding_mode='border', align_corners=True)