import torch

from .sde_vp import VPSchedule


@torch.no_grad()
def sample_with_euler(
    model: torch.nn.Module,
    num_samples: int = 5000,
    num_steps: int = 1000,
    device: torch.device = torch.device("cpu"),
):
    schedule = VPSchedule()
    x = torch.randn(num_samples, 2, device=device)

    # Integrate t from 1 -> 0 with Euler–Maruyama on the reverse SDE
    t_vals = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    dt = -1.0 / num_steps  # negative step

    for k in range(num_steps):
        t = t_vals[k].expand(num_samples)
        beta_t = schedule.beta(t)
        score = model(x, t)

        # Reverse SDE drift for VP: f_rev = -0.5*beta*x - beta*score
        drift = -0.5 * beta_t.view(-1, 1) * x - beta_t.view(-1, 1) * score

        # Diffusion term magnitude for reverse SDE is sqrt(beta)
        noise_scale = torch.sqrt(beta_t.clamp_min(1e-12) * (-dt)).view(-1, 1)
        noise = torch.randn_like(x)

        x = x + dt * drift + noise_scale * noise

    return x.detach()
