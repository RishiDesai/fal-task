import torch
from diffusers import DDPMScheduler


@torch.no_grad()
def sample_with_euler(
        model: torch.nn.Module,
        num_samples: int = 5000,
        num_steps: int = 1000,
        device: torch.device = torch.device("cpu"),
):
    """Sampling with DDPM scheduler to match training (epsilon prediction).

    - Uses the same linear beta schedule as training
    - Keeps the original function name for backward compatibility
    """
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=False,
    )

    scheduler.set_timesteps(num_steps, device=device)
    T = float(scheduler.config.num_train_timesteps - 1)

    x = torch.randn(num_samples, 2, device=device, dtype=torch.float32)
    if hasattr(scheduler, "init_noise_sigma"):
        x = x * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        # Normalize discrete timestep for our time embedding
        t_norm_scalar = float(t.item()) / T
        t_norm = torch.full((num_samples,), t_norm_scalar, device=device, dtype=x.dtype)

        x_in = scheduler.scale_model_input(x, t)
        eps_pred = model(x_in, t_norm)

        step_out = scheduler.step(model_output=eps_pred, timestep=t, sample=x)
        x = step_out.prev_sample

    return x.detach()
