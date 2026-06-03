import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class NumericalAccuracy:
    # Elem-wise relative tolerance. For fp16 this cannot be less than 1e-4 due to precision limits
    rtol: float = 5e-2
    # Elem-wise absolute tolerance. For fp16 this should be close to 2^{-14}, the smallest normalized fp16 number.
    atol: float = 5e-5
    # Frobenius norm-wise relative tolerance (average correct decimal digits). Should be smaller than RTOL.
    ftol: float = 1e-3
    max_rmse_ratio: float = 0.05
    min_r2: float = 0.99
    hard_fail_max: float = 1.0

    def _r2(self, y_ref: torch.Tensor, y_pred: torch.Tensor) -> float:
        ref = y_ref.detach().cpu().numpy().ravel().astype(np.float64)
        pred = y_pred.detach().cpu().numpy().ravel().astype(np.float64)
        ss_res = np.sum((ref - pred) ** 2)
        ss_tot = np.sum((ref - np.mean(ref)) ** 2)
        return float("nan") if ss_tot < 1e-30 else 1.0 - ss_res / ss_tot

    def stats_ok(self, actual: torch.Tensor, expected: torch.Tensor) -> bool:
        diff = (actual - expected).abs()
        mx = diff.max().item()
        if mx > self.hard_fail_max:
            return False
        bound = self.atol + self.rtol * expected.abs()
        if (diff <= bound).all():
            return True
        mean_abs = float(expected.float().flatten().abs().mean())
        rmse = float(torch.sqrt((diff.float().flatten() ** 2).mean()))
        ratio = rmse / max(mean_abs, 1e-15)
        r2 = self._r2(expected, actual)
        if mean_abs < 1e-9:
            return rmse < 5e-4
        return ratio <= self.max_rmse_ratio and np.isfinite(r2) and r2 >= self.min_r2
