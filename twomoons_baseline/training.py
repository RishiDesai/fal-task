
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from diffusers import DDPMScheduler

from .networks import ScoreNet
from .data import make_two_moons


def train_baseline(
    ckpt_path: str,
    num_steps: int = 5000,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    log_every: int = 250,
    grad_clip: float = 1.0,
):
    X, _ = make_two_moons(n_samples=50000)
    X = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    net = ScoreNet().to(device)
    net.train()
    opt = AdamW(net.parameters(), lr=lr)

    # Use a standard DDPM scheduler (discrete-time) with linear betas
    num_train_timesteps = 1000
    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="linear",
        prediction_type="epsilon",
    )

    step = 0
    pbar = tqdm(total=num_steps, desc="training", ncols=100)
    while step < num_steps:
        for (x0_batch,) in loader:
            x0 = x0_batch.to(device)
            b = x0.size(0)
            # Sample discrete diffusion steps
            t_int = torch.randint(0, num_train_timesteps, (b,), device=device, dtype=torch.long)
            # Noise to add and to predict
            eps = torch.randn_like(x0)
            # Forward diffusion using the scheduler
            xt = scheduler.add_noise(x0, eps, t_int)

            # Model predicts epsilon; feed normalized time in [0,1] to the time embedding
            t_norm = t_int.float() / float(num_train_timesteps - 1)
            eps_pred = net(xt, t_norm)

            # Epsilon objective (no extra weighting)
            loss = torch.mean((eps_pred - eps) ** 2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            opt.step()

            step += 1
            if step % log_every == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            pbar.update(1)

            if step >= num_steps:
                break

    pbar.close()

    ckpt = {
        "state_dict": net.state_dict(),
        "config": {"input_dim": 2, "hidden_dim": 128, "time_embed_dim": 64},
    }
    torch.save(ckpt, ckpt_path)
