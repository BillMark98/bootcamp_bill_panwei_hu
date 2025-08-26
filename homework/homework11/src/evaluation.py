import numpy as np
from typing import Tuple, Callable

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Bootstrap CI for a metric on paired (y_true, y_pred).
    Returns (point_estimate, ci_low, ci_high).
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)
    stats = np.empty(n_boot, dtype=float)
    idx = np.arange(n)
    for b in range(n_boot):
        sample_idx = rng.choice(idx, size=n, replace=True)
        stats[b] = metric_fn(y_true[sample_idx], y_pred[sample_idx])
    low = float(np.quantile(stats, alpha/2))
    high = float(np.quantile(stats, 1 - alpha/2))
    return float(point), low, high
