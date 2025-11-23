import torch
import torch.nn as nn
from utils import MVWarp, MVRefiner, ResidualBlock, PartitionMap
class MVSR(nn.Module):
    def __init__(self, mid=64, blocks=15, scale=4):
        super().__init__()
        self.mid = mid
        self.blocks = blocks
        self.scale = scale

        self.mvwarp = MVWarp()
        
        self.feat_extract = nn.Sequential(
            nn.Conv2d(3, mid, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(mid, mid, 3, 1, 1),
            nn.LeakyReLU(0.1, True)
        )
        
        self.mv_refiner = MVRefiner()
        
        self.backward_resblocks = nn.Sequential(*[ResidualBlock(mid) for _ in range(blocks)])
        self.forward_resblocks = nn.Sequential(*[ResidualBlock(mid) for _ in range(blocks)])
        
        self.partion_map = PartitionMap(mid)
        
        self.fusion = nn.Conv2d(mid * 2, mid, 1, 1)
        
        self.up = nn.Sequential(
            nn.Conv2d(mid, mid * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(mid, mid * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(mid, 3, 3, 1, 1)
        )

    def compute_flow(self, mv):
        return self.mv_refiner(mv / 4.0)

    def forward(self, imgs, mv_fwd, mv_bwd, partition_maps):
        B, T, C, H, W = imgs.shape
        
        feats = self.feat_extract(imgs.view(-1, C, H, W)).view(B, T, -1, H, W)
        
        bwd_features = []
        h_bwd = torch.zeros_like(feats[:, 0])
        
        for t in range(T - 1, -1, -1):
            flow = self.compute_flow(mv_bwd[:, t]) 
            h_bwd = self.mvwarp(h_bwd, flow)
            
            h_bwd = h_bwd + feats[:, t]
            h_bwd = self.backward_resblocks(h_bwd)
            bwd_features.append(h_bwd)
        bwd_features = bwd_features[::-1]

        fwd_features = []
        h_fwd = torch.zeros_like(feats[:, 0])
        
        for t in range(T):
            flow = self.compute_flow(mv_fwd[:, t])
            h_fwd = self.mvwarp(h_fwd, flow)
            
            h_fwd = h_fwd + feats[:, t]
            h_fwd = self.forward_resblocks(h_fwd)
            fwd_features.append(h_fwd)

        outs = []
        for t in range(T):
            fused = self.fusion(torch.cat([fwd_features[t], bwd_features[t]], dim=1))
            
            refined = self.partion_map(fused, partition_maps[:, t])
            
            out = self.up(refined)
            outs.append(out)

        return torch.stack(outs, dim=1)
