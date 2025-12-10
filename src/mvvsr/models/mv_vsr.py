import torch
import torch.nn as nn
from src.mvvsr.utils.utils import MVWarp, MVRefiner, ResidualBlock, PartitionMap

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
        self.partion_map = PartitionMap(mid)
        self.backward_resblocks = nn.Sequential(*[ResidualBlock(mid) for _ in range(blocks)])
        self.forward_resblocks = nn.Sequential(*[ResidualBlock(mid) for _ in range(blocks)])
        
        self.refinement = nn.Sequential(
            nn.Conv2d(mid, mid, 3, 1, 1),
            nn.LeakyReLU(0.1, True)
        )
        
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

    def forward(self, imgs, mv_fwd, mv_bwd, frame_types, 
                ablate_second_order=False, ablate_mvs=False):
        B, T, C, H, W = imgs.shape
        if ablate_mvs:
            mv_fwd = torch.zeros_like(mv_fwd)
            mv_bwd = torch.zeros_like(mv_bwd)

        is_p_frame = (frame_types == 1).view(B, T, 1, 1, 1)
        proxy_mv_bwd = -mv_fwd
        mv_bwd = torch.where(is_p_frame, proxy_mv_bwd, mv_bwd)

        feats = self.feat_extract(imgs.view(-1, C, H, W)).view(B, T, -1, H, W)
        
        bwd_features = [None] * T
        h_bwd = torch.zeros_like(feats[:, 0])      
        h_bwd_old = torch.zeros_like(feats[:, 0])  
        
        for t in range(T - 1, -1, -1):
            raw_mv1 = mv_bwd[:, t]
            flow1 = self.compute_flow(raw_mv1)
            h1_warped = self.mvwarp(h_bwd, flow1)
            
            if (not ablate_second_order) and (t < T - 2):
                raw_mv2 = mv_bwd[:, t+1] 
                flow2_base = self.compute_flow(raw_mv2)
                warped_flow2 = self.mvwarp(flow2_base, flow1)
                flow_chain = flow1 + warped_flow2
                h2_warped = self.mvwarp(h_bwd_old, flow_chain)
                
                h_prop = (h1_warped + h2_warped) * 0.5
            else:
                h_prop = h1_warped

            h_bwd_old = h_bwd.clone()
            h_bwd = h_prop + feats[:, t]
            h_bwd = self.backward_resblocks(h_bwd)
            bwd_features[t] = h_bwd


        fwd_features = [None] * T
        h_fwd = torch.zeros_like(feats[:, 0])      
        h_fwd_old = torch.zeros_like(feats[:, 0])
        
        for t in range(T):
            raw_mv1 = mv_fwd[:, t]
            flow1 = self.compute_flow(raw_mv1)
            h1_warped = self.mvwarp(h_fwd, flow1)

            if (not ablate_second_order) and (t > 1):
                raw_mv2 = mv_fwd[:, t-1]
                flow2_base = self.compute_flow(raw_mv2)
                warped_flow2 = self.mvwarp(flow2_base, flow1)
                flow_chain = flow1 + warped_flow2
                h2_warped = self.mvwarp(h_fwd_old, flow_chain)

                h_prop = (h1_warped + h2_warped) * 0.5
            else:
                h_prop = h1_warped
            
            h_fwd_old = h_fwd.clone()
            h_fwd = h_prop + feats[:, t]
            h_fwd = self.forward_resblocks(h_fwd)
            fwd_features[t] = h_fwd

        outs = []
        for t in range(T):
            fused = self.fusion(torch.cat([fwd_features[t], bwd_features[t]], dim=1))            
            refined = self.refinement(fused)
            out = self.up(refined)
            outs.append(out)

        return torch.stack(outs, dim=1)