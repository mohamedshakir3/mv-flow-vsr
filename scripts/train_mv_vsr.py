import os, argparse, yaml
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.mvvsr.models import MVSR
from src.mvvsr.data import RedsMVSRDataset, psnr_torch_perframe, charbonnier_loss


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=torch.float16):
    model.eval()
    psnr_sum, ncount = 0.0, 0
    for imgs, gts, mv_fwd, mv_bwd, partition_maps, ftypes, _, _ in tqdm(loader, desc="val", leave=False):
        imgs     = imgs.to(device, non_blocking=True)       # (B,T,3,H,W)
        gts      = gts.to(device, non_blocking=True)        # (B,T,3,4H,4W)
        mv_fwd   = mv_fwd.to(device, non_blocking=True)     # (B,T,2,H,W)
        mv_bwd   = mv_bwd.to(device, non_blocking=True)     # (B,T,2,H,W)
        partition_maps = partition_maps.to(device, non_blocking=True)
        ftypes   = ftypes.to(device, non_blocking=True)     # (B,T)

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            sr = model(imgs, mv_fwd, mv_bwd, partition_maps, ftypes)

        psnr_sum += float(psnr_torch_perframe(sr, gts)) * imgs.size(0)
        ncount   += imgs.size(0)

    model.train()
    return psnr_sum / max(ncount, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default=None, help="YAML config file")
    ap.add_argument("--num_workers", type=int, default=32)
    ap.add_argument("--flow_tmpl", default="{t:06d}_mv.npz")
    ap.add_argument("--img_tmpl",  default="{:08d}.png")
    ap.add_argument("--out_dir",   default="out_mvsr")
    
    ap.add_argument("--reds_root", type=str, default=None)
    ap.add_argument("--val_reds_root", type=str, default=None)
    ap.add_argument("--flows_root", type=str, default=None)
    ap.add_argument("--val_flows_root", type=str, default=None)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=14)
    ap.add_argument("--crop_lr", type=int, default=96)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ckpt", type=bool, default=False)
    
    args = ap.parse_args()
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f) or {}

    def apply_dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                apply_dict(v)
            else:
                if hasattr(args, k):
                    setattr(args, k, v)

    apply_dict(cfg)

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Datasets / Loaders
    train_ds = RedsMVSRDataset(
            reds_root=args.reds_root,
            codec_root=args.flows_root,
            split="train",
            scale=args.scale,
            seq_len=args.seq_len,
            crop_lr=args.crop_lr,
            augment=True,
            img_tmpl=args.img_tmpl,
    )
    val_ds = RedsMVSRDataset(
        reds_root=args.val_reds_root,
        codec_root=args.val_flows_root,
        split="val",
        scale=args.scale,
        seq_len=args.seq_len,
        crop_lr=None,
        augment=False,
        img_tmpl=args.img_tmpl,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(1, min(2, args.num_workers)), pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )

    model = MVSR(mid=64, blocks=15, scale=args.scale).to(device)
    checkpoint_path = os.path.join(args.out_dir, "best.pth")
    
    if args.ckpt:
        print(f"--- Loading weights from {checkpoint_path} ---")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint, strict=False)
        
        print("--- Weights loaded successfully ---")
    else:
        print(f"--- No checkpoint found at {checkpoint_path}, starting from scratch ---")


    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler()

    os.makedirs(args.out_dir, exist_ok=True)
    best_psnr = -1.0
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    writer = SummaryWriter(log_dir=args.out_dir)
    tb_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}")
        for imgs, gts, mv_fwd, mv_bwd, partition_maps, ftypes, _, _ in pbar:
            imgs           = imgs.to(device, non_blocking=True)
            gts            = gts.to(device, non_blocking=True)
            mv_fwd         = mv_fwd.to(device, non_blocking=True)
            mv_bwd         = mv_bwd.to(device, non_blocking=True)
            partition_maps = partition_maps.to(device, non_blocking=True)
            ftypes   = ftypes.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                sr = model(imgs, mv_fwd, mv_bwd, partition_maps, ftypes)
                loss = charbonnier_loss(sr, gts)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix(
                loss=f"{float(loss.item()):.4f}",
            )
            
            writer.add_scalar("train/loss_total", loss.item(), tb_step)
            tb_step += 1

        val_psnr = evaluate(model, val_loader, device, amp_dtype=amp_dtype)
        print(f"[Epoch: {epoch}] Validation PSNR: {val_psnr:.3f} dB")

        writer.add_scalar("val/psnr", val_psnr, epoch)

        ckpt = os.path.join(args.out_dir, f"epoch_{epoch:04d}.pth")
        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict()}, ckpt)
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pth"))
            print(f"Best PSNR: ({best_psnr:.3f} dB) saved.")

if __name__ == "__main__":
    main()
