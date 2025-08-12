import argparse
import os
from pathlib import Path

import torch

from twomoons_baseline.training import train_baseline
from twomoons_baseline.sampling import sample_with_euler
from twomoons_baseline.metrics import (
    compute_mmd_rbf,
    compute_energy_distance,
    compute_sliced_wasserstein,
    compute_c2st_auc,
)
from twomoons_baseline.utils import set_seed, to_device, plot_generated_vs_target, plot_generated_only, make_two_moons
from twomoons_baseline.networks import ScoreNet


ARTIFACTS_DIR = Path("artifacts")
CKPT_DIR = ARTIFACTS_DIR / "checkpoints"
OUTPUTS_DIR = ARTIFACTS_DIR / "outputs"
PLOTS_DIR = ARTIFACTS_DIR / "plots"


def ensure_dirs():
    for d in [CKPT_DIR, OUTPUTS_DIR, PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _load_model_from_ckpt(ckpt_path: Path, device: torch.device, prefer_ema: bool = True) -> torch.nn.Module:
    # Try to load as a safe state_dict checkpoint
    try:
        ckpt = torch.load(str(ckpt_path), map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            config = ckpt.get("config", {"input_dim": 2, "hidden_dim": 128, "time_embed_dim": 64})
            model = ScoreNet(**config).to(device)
            # Optionally prefer EMA weights if available
            if prefer_ema and "ema_state_dict" in ckpt:
                state = ckpt["ema_state_dict"]
            else:
                state = ckpt["state_dict"]
            model.load_state_dict(state)
            return model
    except Exception:
        pass

    # Fallback for old checkpoints: allowlist ScoreNet for weights_only=True context
    try:
        with torch.serialization.safe_globals([ScoreNet]):
            model = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            return model
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {ckpt_path}: {e}")


def run_baseline(args):
    ensure_dirs()
    device = to_device()
    set_seed(args.seed)

    ckpt_path = CKPT_DIR / "baseline.pt"
    # Train (unless skipped). If skipped but checkpoint is missing, train anyway.
    if not args.skip_train or not ckpt_path.exists():
        train_baseline(
            ckpt_path=str(ckpt_path),
            num_steps=args.train_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            log_every=args.log_every,
            use_ema=args.train_use_ema,
            ema_decay=args.ema_decay,
        )

    # Load model
    model = _load_model_from_ckpt(ckpt_path, device, prefer_ema=args.sample_use_ema)
    model.eval()

    # Sample (Euler sampler)
    with torch.no_grad():
        samples = sample_with_euler(
            model=model,
            num_samples=args.num_samples,
            num_steps=args.sample_steps,
            device=device,
        )
    samples_path = OUTPUTS_DIR / "samples.pt"
    torch.save(samples, str(samples_path))

    # Evaluate
    target, _ = make_two_moons(n_samples=args.num_samples, seed=args.seed)
    target = torch.tensor(target, dtype=torch.float32, device=device)
    mmd = compute_mmd_rbf(samples, target)
    ed = compute_energy_distance(samples, target)
    sw = compute_sliced_wasserstein(samples, target)
    auc, acc = compute_c2st_auc(samples, target)
    print(f"MMD (RBF median heuristic): {mmd.item():.6f}")
    print(f"Energy Distance: {ed.item():.6f}")
    print(f"Sliced Wasserstein-1 (L=256): {sw.item():.6f}")
    print(f"C2ST (logreg) ROC-AUC: {auc:.4f}  Acc: {acc:.4f}  (0.5/0.5 is ideal)")

    # Plot
    fig_path = PLOTS_DIR / "baseline_two_moons.png"
    plot_generated_vs_target(samples.cpu().numpy(), target.cpu().numpy(), str(fig_path))
    print(f"Saved plot to {fig_path}")

    # Save generated-only plot as well
    gen_only_path = PLOTS_DIR / "baseline_two_moons_generated.png"
    plot_generated_only(samples.cpu().numpy(), str(gen_only_path), title="Generated (EMA if available)")
    print(f"Saved generated-only plot to {gen_only_path}")


def main():
    parser = argparse.ArgumentParser(description="Two Moons Diffusion Baseline")
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--sample-steps", type=int, default=1000)
    # EMA controls
    parser.add_argument("--train-use-ema", action="store_true", help="Track EMA during training and save EMA weights")
    parser.add_argument("--sample-use-ema", action="store_true", help="Use EMA weights for sampling if available")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay for training updates")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and only sample/evaluate using existing checkpoint")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run", action="store_true", help="Run train->sample->eval->plot pipeline")
    args = parser.parse_args()

    if args.run:
        run_baseline(args)
    else:
        print("Pass --run to execute the full baseline pipeline.")


if __name__ == "__main__":
    main()
