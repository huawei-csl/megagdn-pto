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

    def stats_ok(
        self, actual: torch.Tensor, expected: torch.Tensor, chunk_size: int = 1
    ) -> bool:
        adjusted_rtol = min(0.5, self.rtol * chunk_size)

        actual_fp64 = actual.double()
        expected_fp64 = expected.double()

        diff = (actual_fp64 - expected_fp64).abs()
        frob_rel_error = torch.sqrt(torch.sum(diff**2) / torch.sum(expected_fp64**2))
        rel_err_bound = self.atol + adjusted_rtol * expected_fp64.abs()
        if (diff > rel_err_bound).all():
            print(
                f"ERROR: max relative error larger than the bound: {(diff).max().item()}. ATOL={self.atol} RTOL={adjusted_rtol}"
            )
            return False
        if frob_rel_error > self.ftol:
            print(
                f"ERROR: large frobenius relative error: {frob_rel_error}. FTOL={self.ftol}"
            )
            return False
        return True


def generate_random_inputs(T, H, HG, D):
    q = F.normalize(torch.randn(1, T, HG, D, dtype=torch.float16), dim=-1, p=2)
    k = F.normalize(torch.randn(1, T, HG, D, dtype=torch.float16), dim=-1, p=2)
    v = torch.randn(1, T, H, D, dtype=torch.float16)
    beta = torch.rand(1, T, H, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(1, T, H, dtype=torch.float32))
    return q, k, v, beta, g_in
