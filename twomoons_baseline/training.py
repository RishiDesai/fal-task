
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .networks import ScoreNet
from .sde_vp import VPSchedule
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

    schedule = VPSchedule()

    step = 0
    pbar = tqdm(total=num_steps, desc="training", ncols=100)
    while step < num_steps:
        for (x0_batch,) in loader:
            x0 = x0_batch.to(device)
            b = x0.size(0)
            t = torch.rand(b, device=device)  # U[0,1]

            alpha_t = schedule.alpha(t)
            sigma_t = schedule.sigma(t)

            z = torch.randn_like(x0)
            xt = alpha_t.view(-1, 1) * x0 + sigma_t.view(-1, 1) * z

            # Target score: grad_x log p(xt|x0) = (mu - x)/sigma^2
            sigma2_t = (sigma_t * sigma_t).view(-1, 1)
            target_score = (alpha_t.view(-1, 1) * x0 - xt) / (sigma2_t + 1e-12)

            pred = net(xt, t)

            # Weighting lambda(t) = g(t)^2. For VP-SDE, g(t)^2 = beta(t)
            beta_t = schedule.beta(t).view(-1, 1)
            weight = beta_t
            loss = (weight * (pred - target_score) ** 2).mean()

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
