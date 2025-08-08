import torch
from diffusers import DDPMScheduler


@torch.no_grad()
def sample_with_euler(
    model: torch.nn.Module,
    num_samples: int = 5000,
    num_steps: int = 1000,
    device: torch.device = torch.device("cpu"),
):
    """Euler–Maruyama on the VP reverse SDE using DDPM discrete statistics.

    We use the scheduler's alphas_cumprod to compute per-step beta(t) and
    convert the model's epsilon prediction into a score for the SDE drift.
    """
    # Use a DDPM scheduler to obtain discrete betas/alphas_cumprod
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="linear",
        prediction_type="epsilon",
    )

    x = torch.randn(num_samples, 2, device=device)
    # Use the scheduler to define a sequence of discrete indices
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps.to(device)  # descending ints of length num_steps
    dt = -1.0 / num_steps  # continuous step for SDE integration

    # Precompute alphas_cumprod on the same device (length = num_train_timesteps)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    for k, t_idx in enumerate(timesteps):
        # Discrete stats at current step
        t_idx_int = t_idx.long().clamp(0, len(alphas_cumprod) - 1)
        at = alphas_cumprod[t_idx_int]
        # Instantaneous variance growth approximated from discrete cumulative alpha
        # sigma_t^2 = 1 - a_t, where a_t = alpha_cumprod
        sigma_t = torch.sqrt(torch.clamp(1.0 - at, min=1e-12))

        # Predict epsilon then convert to score s_theta(x,t) = -(x - sqrt(a_t) x0)/sigma_t^2
        # Using x = sqrt(a_t) x0 + sigma_t eps => x0 = (x - sigma_t eps)/sqrt(a_t)
        # Implies score = - (x - sqrt(a_t) x0)/sigma_t^2 = - (x - (x - sigma_t eps))/sigma_t^2 = - eps / sigma_t
        t_norm_batch = (t_idx_int.float() / float(len(alphas_cumprod) - 1)).expand(num_samples)
        eps_pred = model(x, t_norm_batch)
        score = -eps_pred / (sigma_t + 1e-12)

        # Approximate instantaneous beta(t) from discrete schedule:
        # beta_eff ≈ - d/dt log a(t). Use finite difference over indices.
        if k < len(timesteps) - 1:
            idx_next = timesteps[k + 1].long().clamp(0, len(alphas_cumprod) - 1)
        else:
            idx_next = torch.clamp(t_idx_int - 1, min=0)
        a_curr = alphas_cumprod[t_idx_int]
        a_next = alphas_cumprod[idx_next]
        # Ensure positivity and stability
        a_curr = torch.clamp(a_curr, min=1e-12)
        a_next = torch.clamp(a_next, min=1e-12)
        # Approximate derivative over the actual dt (note: dt < 0 when integrating 1->0)
        beta_t = torch.clamp(-(torch.log(a_next) - torch.log(a_curr)) / (dt), min=1e-8)

        drift = -0.5 * beta_t * x - beta_t * score
        noise_scale = torch.sqrt(beta_t * (-dt))
        noise = torch.randn_like(x)
        x = x + dt * drift + noise_scale * noise

    return x.detach()
