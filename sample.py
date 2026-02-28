# python sample.py --name ACMDM_Flow_S_PatchSize22 --text_prompt "A person is running on a treadmill." --motion_length 196 --cfg 3.0 --save_mp4

import os
from os.path import join as pjoin
import argparse
import re
import numpy as np
import torch

from models.AE_2D_Causal import AE_models
from models.ACMDM import ACMDM_models

# Reuse the exact plotting util from MARDM
# Assumes your repo layout matches ../MARDM/utils/motion_process.py
from utils.motion_process import plot_3d_motion, kit_kinematic_chain, t2m_kinematic_chain


def safe_stem(s: str, max_len: int = 120) -> str:
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("._-")
    return s or "caption"


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
    if unexpected:
        raise RuntimeError(f"Unexpected keys in ACMDM checkpoint: {unexpected}")
    if any((not k.startswith("clip_model.")) for k in missing):
        raise RuntimeError(f"Missing non-CLIP keys in ACMDM checkpoint: {missing}")
    acmdm.eval().to(device)

    # --- AE post mean/std for latent unnormalization
    after_mean = np.load(pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, "AE_2D_Causal_Post_Mean.npy"))
    after_std  = np.load(pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, "AE_2D_Causal_Post_Std.npy"))

    # --- Length handling (AE downsamples time by 4)
    motion_len = int(args.motion_length)
    latent_len = max(1, motion_len // 4)

    # --- Generate latents -> decode -> unnormalize to xyz
    with torch.no_grad():
        m_lens = torch.tensor([latent_len], device=device, dtype=torch.long)

        latents = acmdm.generate(
            conds=[args.text_prompt],    # list[str]
            m_lens=m_lens,              # torch.LongTensor([latent_len])
            cond_scale=float(args.cfg),
            j=22
        )  # (B, C, latent_len, 22)

        # Unnormalize latents: (B,C,L,J) -> (B,L,J,C) for broadcast with after_{mean,std} (C,)
        lat_np = latents.permute(0, 2, 3, 1).cpu().numpy()     # (B,L,J,C)
        lat_np = lat_np * after_std + after_mean               # broadcast on last dim
        lat_t  = torch.from_numpy(lat_np).to(device).permute(0, 3, 1, 2)  # (B,C,L,J)

        # Decode to motion (normalized 22x3), then unnormalize to xyz
        motion_norm = ae.decode(lat_t).detach().cpu().numpy()
        # expected: (B, T, J, 3) (or similar); your code assumes this and broadcasts mean/std below

        mean = mean_22x3.reshape(1, 1, 1, 3)
        std  = std_22x3.reshape(1, 1, 1, 3)
        motion_xyz = motion_norm * std + mean   # (B, T, J, 3)

        # Save format you already used: (B, 3, T, J)
        motion_xyz_b3tj = np.transpose(motion_xyz, (0, 3, 1, 2))

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Save .npy in MARDM format: (T, J, 3), float32
    out_npy_path = pjoin(args.out_dir, args.out_name)

    # motion_xyz_b3tj[0] is (3, T, J) -> (T, J, 3)
    joint_tj3 = np.transpose(motion_xyz_b3tj[0], (1, 2, 0))

    # Trim to requested length exactly (like MARDM does with joint_data[:ml])
    joint_tj3 = joint_tj3[:motion_len]

    # Match MARDM dtype
    joint_tj3 = joint_tj3.astype(np.float32, copy=False)

    np.save(out_npy_path, joint_tj3)
    print(f"Saved NPY: {out_npy_path}, shape={joint_tj3.shape} (T,J,3)")

    # --- Plot MP4 like ../MARDM/sample.py
    if args.save_mp4:
        # Convert (3,T,J) -> (T,J,3) for plot_3d_motion
        joints_tj3 = np.transpose(motion_xyz_b3tj[0], (1, 2, 0))  # (T, J, 3)

        # Optionally trim to requested length (in case AE decode differs slightly)
        joints_tj3 = joints_tj3[:motion_len]

        kinematic_chain = kit_kinematic_chain if args.dataset_name == "kit" else t2m_kinematic_chain
        cap_stem = safe_stem(args.text_prompt)

        mp4_name = args.mp4_name
        if mp4_name == "":
            mp4_name = f"{cap_stem}_len{motion_len}_cfg{args.cfg}.mp4"

        out_mp4_path = pjoin(args.out_dir, mp4_name)
        plot_3d_motion(
            out_mp4_path,
            kinematic_chain,
            joints_tj3,
            title=args.text_prompt,
            fps=int(args.fps),
            radius=float(args.radius),
        )
        print(f"Saved MP4: {out_mp4_path}")


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

    # New plotting args
    p.add_argument("--save_mp4", action="store_true", help="If set, also save an MP4 using plot_3d_motion.")
    p.add_argument("--mp4_name", type=str, default="", help="Optional mp4 filename (default: derived from prompt).")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--radius", type=float, default=4.0)

    args = p.parse_args()
    main(args)