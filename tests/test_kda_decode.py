"""End-to-end validation of the fused KDA decode kernel on real NPU.

Chains nothing -- ONE launch of kdn_decode.so computes o and S_out
in a single call, then compares against naive_recurrent_kda (the ORIGINAL torch
source) with the reference computed on CPU in float64.

The fla package needs triton, so naive.py is loaded directly via importlib
(same pattern as verify_chain_e2e.py) rather than importing the package.
"""
import argparse
import ctypes
import importlib.util
import os
import sys

import torch
import torch_npu  # noqa: F401

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_SO = os.path.join(_REPO, "kernels", "pto", "compiled_lib", "kdn_decode.so")

# The fp64 reference is the ORIGINAL torch source `naive_recurrent_kda` from
# flash-linear-attention (fla/ops/kda/naive.py). fla pulls in triton, so the file
# is loaded directly rather than importing the package. Point FLA_KDA_NAIVE at it.
NAIVE = os.environ.get("FLA_KDA_NAIVE", "")
if not NAIVE or not os.path.exists(NAIVE):
    raise SystemExit(
        "Set FLA_KDA_NAIVE to flash-linear-attention/fla/ops/kda/naive.py "
        "(the naive_recurrent_kda reference)."
    )

spec = importlib.util.spec_from_file_location("kda_naive", NAIVE)
_m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_m)
naive_recurrent_kda = _m.naive_recurrent_kda


def load(so_path):
    lib = ctypes.CDLL(os.path.realpath(so_path))
    fn = lib.call_kernel
    fn.argtypes = [
        ctypes.c_uint32,   # block_dim
        ctypes.c_void_p,   # stream
        ctypes.c_void_p,   # S_in
        ctypes.c_void_p,   # q
        ctypes.c_void_p,   # k
        ctypes.c_void_p,   # v
        ctypes.c_void_p,   # g
        ctypes.c_void_p,   # beta
        ctypes.c_void_p,   # S_out
        ctypes.c_void_p,   # o
        ctypes.c_int64,    # total_work
        ctypes.c_int64,    # K
        ctypes.c_int64,    # V
    ]
    fn.restype = None
    return fn


def bd(total_work):
    # Cap at 47: a subblock-guarded vector kernel schedules at most 47 blocks per
    # wave on 910B2 (block_dim=48 spills to a 2nd wave). Correctness is unaffected
    # (grid-stride), but 47 avoids the cliff.
    return max(1, min(int(total_work), 47))


def rel(a, b):
    a = a.double()
    b = b.double()
    d = torch.linalg.norm((a - b).flatten())
    n = torch.linalg.norm(b.flatten())
    return (d / n).item() if n > 0 else d.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel_so", nargs="?", default=_DEFAULT_SO)
    ap.add_argument("--b-list", type=int, nargs="+", default=[1, 4, 16, 32])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--hv", type=int, default=16)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--tol", type=float, default=1e-5)
    a = ap.parse_args()

    torch.npu.set_device(0)
    stream = ctypes.c_void_p(torch_npu.npu.current_stream().npu_stream)
    fn = load(a.kernel_so)

    HV, K = a.hv, a.k
    V = K
    H = HV  # G = 1
    P = lambda t: ctypes.c_void_p(t.contiguous().data_ptr())
    I = ctypes.c_int64

    worst_o = 0.0
    worst_S = 0.0
    npass = 0
    nfail = 0

    for B in a.b_list:
        for seed in a.seeds:
            torch.manual_seed(seed)
            # host fp32 inputs, T=1
            q = torch.rand(B, 1, H, K)
            k = torch.rand(B, 1, H, K)
            v = torch.rand(B, 1, HV, V)
            g = torch.nn.functional.logsigmoid(torch.randn(B, 1, HV, K))
            beta = torch.randn(B, 1, HV).sigmoid()
            h0 = torch.randn(B, HV, K, V)

            # ---- CPU float64 reference via the ORIGINAL algorithm
            o_ref, S_ref = naive_recurrent_kda(
                q=q.double(), k=k.double(), v=v.double(), g=g.double(),
                beta=beta.double(), initial_state=h0.double(),
                output_final_state=True)
            o_ref = o_ref[:, 0]

            # ---- NPU fused kernel (fp32), single launch
            d = lambda t: t.to('npu').contiguous()
            S_in = d(h0)
            qn = d(q[:, 0])
            kn = d(k[:, 0])
            vn = d(v[:, 0])
            gn = d(g[:, 0])
            bn = d(beta[:, 0])
            S_out = torch.zeros_like(S_in)
            o = torch.zeros(B, HV, V, dtype=torch.float32, device='npu')
            tw = B * HV

            torch.npu.synchronize()
            fn(ctypes.c_uint32(bd(tw)), stream,
               P(S_in), P(qn), P(kn), P(vn), P(gn), P(bn),
               P(S_out), P(o), I(tw), I(K), I(V))
            torch.npu.synchronize()

            eo = rel(o.cpu(), o_ref)
            eS = rel(S_out.cpu(), S_ref)
            worst_o = max(worst_o, eo)
            worst_S = max(worst_S, eS)
            ok = eo < a.tol and eS < a.tol
            npass += ok
            nfail += (not ok)
            print("B=%-3d seed=%d  o_rel=%.4e  S_rel=%.4e  %s"
                  % (B, seed, eo, eS, "PASS" if ok else "FAIL"))

    print("\nworst o_rel_err = %.4e  (tol %.1e)" % (worst_o, a.tol))
    print("worst S_rel_err = %.4e  (tol %.1e)" % (worst_S, a.tol))
    print("cases: %d PASS, %d FAIL" % (npass, nfail))
    print("E2E FUSED:", "PASS" if nfail == 0 else "FAIL")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
