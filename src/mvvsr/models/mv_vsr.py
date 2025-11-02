import torch.nn.functional as F

def downsample_flow(flow, scale):
    ds_flow = F.avg_pool2d(flow, kernel_size=int(1/scale), stride=int(1/scale))
    return ds_flow * scale


