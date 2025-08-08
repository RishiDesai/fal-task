from typing import Optional

import torch


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    # x: (n, d), y: (m, d)
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True)
    dist2 = x_norm + y_norm.t() - 2.0 * x @ y.t()
    k = torch.exp(-gamma * dist2)
    return k


def _median_heuristic_bandwidth(x: torch.Tensor, y: torch.Tensor) -> float:
    z = torch.cat([x, y], dim=0)
    with torch.no_grad():
        dist2 = torch.cdist(z, z) ** 2
        med = torch.median(dist2[dist2 > 0])
        sigma2 = med.item() / (2.0 + 1e-12)
        sigma2 = max(sigma2, 1e-6)
    return 1.0 / sigma2


def compute_mmd_rbf(x: torch.Tensor, y: torch.Tensor, num_scales: int = 5) -> torch.Tensor:
    # Use multiple bandwidths around the median heuristic
    gamma0 = _median_heuristic_bandwidth(x, y)
    gammas = [gamma0 * (2.0 ** i) for i in range(-(num_scales // 2), (num_scales // 2) + 1)]

    m = x.size(0)
    n = y.size(0)

    mmd2_total = 0.0
    for gamma in gammas:
        Kxx = _rbf_kernel(x, x, gamma)
        Kyy = _rbf_kernel(y, y, gamma)
        Kxy = _rbf_kernel(x, y, gamma)

        # Unbiased estimator
        sum_Kxx = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1) + 1e-12)
        sum_Kyy = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1) + 1e-12)
        sum_Kxy = Kxy.mean()

        mmd2 = sum_Kxx + sum_Kyy - 2.0 * sum_Kxy
        mmd2_total = mmd2_total + mmd2

    mmd = torch.sqrt(torch.clamp(mmd2_total / len(gammas), min=0.0))
    return mmd


def compute_energy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Energy distance between two point sets.

    ED(X, Y) = sqrt( max( 2 E||X-Y|| - E||X-X'|| - E||Y-Y'||, 0 ) )
    """
    # Means of pairwise distances
    exy = torch.cdist(x, y).mean()
    exx = torch.cdist(x, x).mean()
    eyy = torch.cdist(y, y).mean()
    ed2 = 2.0 * exy - exx - eyy
    return torch.sqrt(torch.clamp(ed2, min=0.0))


def compute_sliced_wasserstein(
    x: torch.Tensor,
    y: torch.Tensor,
    num_projections: int = 256,
) -> torch.Tensor:
    """Sliced Wasserstein-1 distance via random projections.

    Projects to 1D along random unit directions and averages 1D W1.
    """
    device = x.device
    dim = x.size(1)
    directions = torch.randn(num_projections, dim, device=device)
    directions = directions / (directions.norm(dim=1, keepdim=True) + 1e-12)

    x_proj = x @ directions.T  # (n, L)
    y_proj = y @ directions.T  # (m, L)

    sw_total = 0.0
    for i in range(num_projections):
        xi = x_proj[:, i].sort().values
        yi = y_proj[:, i].sort().values
        # Match by quantiles (assumes equal sizes; if different, interpolate indices)
        if xi.numel() == yi.numel():
            sw_total += (xi - yi).abs().mean()
        else:
            # Linear interpolate to the smaller size
            n = min(xi.numel(), yi.numel())
            idx = torch.linspace(0, 1, steps=n, device=device)
            xi_q = torch.quantile(xi, idx)
            yi_q = torch.quantile(yi, idx)
            sw_total += (xi_q - yi_q).abs().mean()

    return sw_total / num_projections
