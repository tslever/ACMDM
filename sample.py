import os
from os.path import join as pjoin
import argparse
import numpy as np
import torch

from models.AE_2D_Causal import AE_models
from models.ACMDM import ACMDM_models


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load 22x3 mean/std used to normalize joint coords during training
    mean_22x3 = np.load(f"utils/22x3_mean_std/{args.dataset_name}/22x3_mean.npy")
    std_22x3  = np.load(f"utils/22x3_mean_std/{args.dataset_name}/22x3_std.npy")

    # --- Load AE
    ae = AE_models[args.ae_model](input_width=3)  # HumanML3D uses xyz per joint
    ae_ckpt = torch.load(
        pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, "model", "latest.tar"),
        map_location="cpu",
    )
    ae.load_state_dict(ae_ckpt["ae"])
    ae.eval().to(device)

    # --- Load ACMDM (EMA weights)
    acmdm = ACMDM_models[args.model](input_dim=ae.output_emb_width, cond_mode="text")
    acmdm_ckpt_path = pjoin(args.checkpoints_dir, args.dataset_name, args.name, "model", "latest.tar")
    acmdm_ckpt = torch.load(acmdm_ckpt_path, map_location="cpu")
    missing, unexpected = acmdm.load_state_dict(acmdm_ckpt["ema_acmdm"], strict=False)
    # CLIP weights are expected to be "missing" because they are created at init time
    if unexpected:
        raise RuntimeError(f"Unexpected keys in ACMDM checkpoint: {unexpected}")
    if any((not k.startswith("clip_model.")) for k in missing):
        raise RuntimeError(f"Missing non-CLIP keys in ACMDM checkpoint: {missing}")

    acmdm.eval().to(device)

    # --- AE post mean/std for latent unnormalization (used in eval code path)
    after_mean = np.load(pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, "AE_2D_Causal_Post_Mean.npy"))
    after_std  = np.load(pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, "AE_2D_Causal_Post_Std.npy"))

    # --- Length handling:
    # In ACMDM eval they pass m_length//4 when using AE latents (because AE downsamples time by 4).
    # So if you want e.g. 196 frames, latent_len should be 49.
    motion_len = int(args.motion_length)
    latent_len = max(1, motion_len // 4)

    # --- Generate latents
    with torch.no_grad():
        m_lens = torch.tensor([latent_len], device = device, dtype = torch.long)
        latents = acmdm.generate(
            conds=[args.text_prompt],     # list[str]
            m_lens=m_lens,          # list[int]
            cond_scale=float(args.cfg),
            j=22
        )  # (B, C, latent_len, 22)

        # Unnormalize latents: latents are (B,C,L,J) but mean/std are for last dim = C
        lat_np = latents.permute(0, 2, 3, 1).cpu().numpy()     # (B,L,J,C)
        lat_np = lat_np * after_std + after_mean               # inv_transform(..., after_mean, after_std)
        lat_t  = torch.from_numpy(lat_np).to(device).permute(0, 3, 1, 2)  # (B,C,L,J)

        # Decode to motion (normalized 22x3), then unnormalize to xyz
        motion_norm = ae.decode(lat_t).detach().cpu().numpy()
        print("decode shape:", motion_norm.shape)

        mean = mean_22x3.reshape(1, 1, 1, 3)
        std = std_22x3.reshape(1, 1, 1, 3)
        motion_xyz = motion_norm * std + mean
        print("mean/std shapes:", mean_22x3.shape, std_22x3.shape)

        motion_xyz = np.transpose(motion_xyz, (0, 3, 1, 2))

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = pjoin(args.out_dir, args.out_name)
    np.save(out_path, motion_xyz[0])
    print(f"Saved: {out_path}, shape={motion_xyz[0].shape}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="Run folder name under checkpoints/<dataset>/<name>/model/latest.tar")
    p.add_argument("--model", default="ACMDM-Flow-S-PatchSize22")
    p.add_argument("--ae_name", default="AE_2D_Causal")
    p.add_argument("--ae_model", default="AE_Model")
    p.add_argument("--dataset_name", default="t2m")
    p.add_argument("--checkpoints_dir", default="./checkpoints")

    p.add_argument("--text_prompt", required=True)
    p.add_argument("--motion_length", type=int, default=196, help="Desired output length in frames (approx).")
    p.add_argument("--cfg", type=float, default=3.0)

    p.add_argument("--out_dir", default="./outputs")
    p.add_argument("--out_name", default="sample_motion.npy")
    args = p.parse_args()
    main(args)