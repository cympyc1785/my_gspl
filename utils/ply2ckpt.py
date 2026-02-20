# ply2ckpt.py
import add_pypath  # noqa: F401
import os
import argparse
import torch
import lightning  # noqa: F401

from internal.utils.gaussian_utils import GaussianPlyUtils


def ensure_shs_rest_shape(shs_rest: torch.Tensor, target_k: int) -> torch.Tensor:
    assert shs_rest.ndim == 3 and shs_rest.shape[-1] == 3
    n, k, _ = shs_rest.shape
    if k == target_k:
        return shs_rest
    if k == 0 and target_k > 0:
        return torch.zeros((n, target_k, 3), dtype=shs_rest.dtype, device=shs_rest.device)
    if k > target_k:
        return shs_rest[:, :target_k, :]
    pad = torch.zeros((n, target_k - k, 3), dtype=shs_rest.dtype, device=shs_rest.device)
    return torch.cat([shs_rest, pad], dim=1)


def build_state_dict_for_gspl(gaussian: GaussianPlyUtils, active_sh_degree: int) -> dict:
    assert isinstance(gaussian.xyz, torch.Tensor)
    p = "gaussian_model.gaussians."
    n = gaussian.xyz.shape[0]

    sd = {
        f"{p}means": gaussian.xyz.contiguous(),
        f"{p}opacities": gaussian.opacities.contiguous(),
        f"{p}shs_dc": gaussian.features_dc.contiguous(),
        f"{p}shs_rest": gaussian.features_rest.contiguous(),
        f"{p}scales": gaussian.scales.contiguous(),
        f"{p}rotations": gaussian.rotations.contiguous(),
        "gaussian_model._active_sh_degree": torch.tensor(int(active_sh_degree), dtype=torch.int64, device=gaussian.xyz.device),

        # required by your earlier KeyError (and not flagged as unexpected)
        "density_controller.max_radii2D": torch.zeros((n,), dtype=torch.float32, device=gaussian.xyz.device),
        "density_controller.xyz_gradient_accum": torch.zeros((n, 1), dtype=torch.float32, device=gaussian.xyz.device),
        "density_controller.denom": torch.zeros((n, 1), dtype=torch.float32, device=gaussian.xyz.device),
    }
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Input gaussian .ply path")
    ap.add_argument("--output", "-o", default=None)
    ap.add_argument("--override", action="store_true", default=False)
    ap.add_argument("--sh-degree", type=int, default=-2,
                    help="-2=auto from ply, 0..3=force")
    ap.add_argument("--active-sh-degree", type=int, default=3,
                    help="Value for gaussian_model._active_sh_degree (default: 3)")
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    args = ap.parse_args()

    ply_path = args.input
    assert ply_path.lower().endswith(".ply"), f"Input must be .ply, got: {ply_path}"

    out_path = args.output or (ply_path[: ply_path.rfind(".")] + ".ckpt")
    if (not args.override) and os.path.exists(out_path):
        raise FileExistsError(f"Output exists: {out_path} (use --override)")

    device = torch.device(args.device)

    # Load from ply
    if args.sh_degree == -2:
        gaussian_np = GaussianPlyUtils.load_from_ply(ply_path, sh_degrees=-1)  # auto
        ply_sh_degree = int(gaussian_np.sh_degrees)
    else:
        gaussian_np = GaussianPlyUtils.load_from_ply(ply_path, sh_degrees=int(args.sh_degree))
        ply_sh_degree = int(args.sh_degree)

    gaussian = gaussian_np.to_parameter_structure()
    for f in ["xyz", "opacities", "features_dc", "features_rest", "scales", "rotations"]:
        setattr(gaussian, f, getattr(gaussian, f).to(device))

    # Your current model expects degree=3 => K=15 (per your error log)
    expected_k = 15
    gaussian.features_rest = ensure_shs_rest_shape(gaussian.features_rest, expected_k)

    n = int(gaussian.xyz.shape[0])
    print(f"PLY loaded: N={n}, ply_sh_degree={ply_sh_degree}, ckpt_active_sh_degree={args.active_sh_degree}, restK={gaussian.features_rest.shape[1]}")

    sd = build_state_dict_for_gspl(gaussian, active_sh_degree=args.active_sh_degree)

    ckpt = {
        "state_dict": sd,
        "pytorch-lightning_version": getattr(lightning, "__version__", "unknown"),
        "global_step": int(args.iteration),
        "epoch": 0,

        # ---- IMPORTANT: prevent Lightning from KeyError when it tries to restore optimizers/schedulers ----
        "optimizer_states": [],
        "lr_schedulers": [],
        "callbacks": {},
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
