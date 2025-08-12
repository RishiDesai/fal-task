import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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


def compute_c2st_auc(
        gen: torch.Tensor,
        real: torch.Tensor,
        test_size: float = 0.3,
        seed: int = 42,
        classifier: str = "rbf_svm",
        n_splits: int = 5,
        n_repeats: int = 3,
):
    """
    Classifier Two-Sample Test (C2ST).

    Defaults to a non-linear classifier (RBF-SVM) and evaluates with
    Repeated Stratified K-Fold cross-validation, which is more reliable
    for nonlinearly separable distributions like two moons.

    Args:
        gen: Generated samples, shape (N, D)
        real: Real samples, shape (M, D)
        test_size: Kept for backward compatibility (unused when CV is used)
        seed: RNG seed
        classifier: "rbf_svm" (default) or "logreg"
        n_splits: Number of CV folds
        n_repeats: Number of CV repeats

    Returns:
        (roc_auc_mean: float, acc_mean: float)
    """
    # Ensure CPU numpy arrays for sklearn
    real_np = real.detach().cpu().numpy()
    gen_np = gen.detach().cpu().numpy()

    # Balance classes by subsampling to the smaller size
    n = min(len(real_np), len(gen_np))
    if len(real_np) != n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(real_np), size=n, replace=False)
        real_np = real_np[idx]
    if len(gen_np) != n:
        rng = np.random.default_rng(seed + 1)
        idx = rng.choice(len(gen_np), size=n, replace=False)
        gen_np = gen_np[idx]

    X = np.concatenate([real_np, gen_np], axis=0)
    y = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)], axis=0)

    # Classifier factory
    if classifier == "rbf_svm":
        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", gamma="scale", C=1.0, probability=True, random_state=seed),
        )
    elif classifier == "logreg":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed),
        )
    else:
        raise ValueError(f"Unknown classifier '{classifier}'")

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    aucs: list[float] = []
    accs: list[float] = []
    for tr_idx, te_idx in cv.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        clf.fit(X_tr, y_tr)

        # Prefer probabilities; fall back to decision scores for AUC if needed
        if hasattr(clf[-1], "predict_proba"):
            prob = clf.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score(y_te, prob))
        else:
            score = clf.decision_function(X_te)
            aucs.append(roc_auc_score(y_te, score))

        pred = clf.predict(X_te)
        accs.append(accuracy_score(y_te, pred))

    return float(np.mean(aucs)), float(np.mean(accs))
