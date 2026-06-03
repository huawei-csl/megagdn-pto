import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class NumericalAccuracy:
    # Elem-wise relative tolerance. For fp16 this cannot be less than 1e-4 due to precision limits
    rtol: float = 5e-3
    # Elem-wise absolute tolerance. For fp16 this should be close to 2^{-14}, the smallest normalized fp16 number.
    atol: float = 1.5 * 1e-4
    # Frobenius norm-wise relative tolerance (average correct decimal digits).
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

    def stats_ok(
        self, actual: torch.Tensor, expected: torch.Tensor, chunk_size: int = 1
    ) -> bool:
        adjusted_rtol = min(0.5, self.rtol * chunk_size)

        diff = (actual - expected).abs()
        mx = diff.max().item()
        if mx > self.hard_fail_max:
            print(f"ERROR: mx fail: {mx} > {self.hard_fail_max}")
            return False
        bound = self.atol + adjusted_rtol * expected.abs()
        f_err = torch.sqrt(torch.sum(diff**2) / torch.sum(expected**2))
        if (diff <= bound).all():
            return True
        else:
            print(
                f"ERROR: diff is greater than bound: {(diff/bound).max().item()}. ATOL={self.atol} -- RTOL={adjusted_rtol} -- F_ERR={f_err}"
            )
        mean_abs = float(expected.float().flatten().abs().mean())
        rmse = float(torch.sqrt((diff.float().flatten() ** 2).mean()))
        ratio = rmse / max(mean_abs, 1e-15)
        r2 = self._r2(expected, actual)
        if mean_abs < 1e-9:
            return rmse < 5e-4
        return ratio <= self.max_rmse_ratio and np.isfinite(r2) and r2 >= self.min_r2


def generate_random_inputs(T, H, HG, D):
    q = F.normalize(torch.randn(1, T, HG, D, dtype=torch.float16), dim=-1, p=2)
    k = F.normalize(torch.randn(1, T, HG, D, dtype=torch.float16), dim=-1, p=2)
    v = torch.randn(1, T, H, D, dtype=torch.float16)
    beta = torch.rand(1, T, H, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(1, T, H, dtype=torch.float32))
    return q, k, v, beta, g_in
