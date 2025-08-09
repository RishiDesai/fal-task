import torch
from diffusers import EulerDiscreteScheduler


@torch.no_grad()
def sample_with_euler(
    model: torch.nn.Module,
    num_samples: int = 5000,
    num_steps: int = 1000,
    device: torch.device = torch.device("cpu"),
):
    """Sampling with diffusers' EulerDiscreteScheduler (epsilon prediction).

    - Matches training schedule (DDPM linear betas, 1000 steps)
    - Uses scheduler.scale_model_input as required by EulerDiscrete
    """
    scheduler = EulerDiscreteScheduler()

    scheduler.set_timesteps(num_steps, device=device)
    T = float(scheduler.config.num_train_timesteps - 1)

    x = torch.randn(num_samples, 2, device=device, dtype=torch.float32)
    # Scale initial noise to scheduler's init sigma for correct trajectory
    if hasattr(scheduler, "init_noise_sigma"):
        x = x * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        # Normalize discrete timestep for our time embedding
        t_norm_scalar = float(t.item()) / T
        t_norm = torch.full((num_samples,), t_norm_scalar, device=device, dtype=x.dtype)

        # Proper usage for EulerDiscreteScheduler
        x_in = scheduler.scale_model_input(x, t)
        eps_pred = model(x_in, t_norm)

        step_out = scheduler.step(model_output=eps_pred, timestep=t, sample=x)
        x = step_out.prev_sample

    return x.detach()
