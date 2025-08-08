from typing import Tuple

import torch


class VPSchedule:
    def __init__(self, beta_min: float = 0.1, beta_max: float = 10.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        # Integral_0^t beta(s) ds = beta_min * t + 0.5 * (beta_max - beta_min) * t^2
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t * t

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        # alpha(t) = exp(-0.5 * integral beta)
        return torch.exp(-0.5 * self.integral_beta(t))

    def sigma2(self, t: torch.Tensor) -> torch.Tensor:
        a = self.alpha(t)
        return 1.0 - a * a

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.sigma2(t).clamp_min(1e-12))
