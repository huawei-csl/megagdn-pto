#!/usr/bin/env python3
"""Benchmark for the fused KDA decode kernel (single launch).

Device-side torch.npu.Event timing with a 256 MiB int8 scratch flush before
EVERY timed call (same method as the per-stage benchmarks). Emits bytes_moved
and achieved GB/s.

bytes_moved per work item (K=V=128, fp32):
    read  S_in  K*V*4
    read  q,k,g 3*K*4
    read  v     V*4
    read  beta  4
    write S_out K*V*4
    write o     V*4
"""
import argparse
import ctypes
import json
import os
import statistics

import torch
import torch_npu  # noqa: F401

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_SO = os.path.join(_REPO, "kernels", "pto", "compiled_lib", "kdn_decode.so")


def load_kernel(so_path):
    lib = ctypes.CDLL(os.path.abspath(so_path))
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


def call(fn, stream_ptr, S_in, q, k, v, g, beta, S_out, o, tw, K, V, bd):
    P = lambda t: ctypes.c_void_p(t.data_ptr())
    fn(ctypes.c_uint32(bd), stream_ptr, P(S_in), P(q), P(k), P(v), P(g),
       P(beta), P(S_out), P(o),
       ctypes.c_int64(int(tw)), ctypes.c_int64(int(K)), ctypes.c_int64(int(V)))


def bytes_moved(total_work, K, V):
    per = (K * V * 4       # read S_in
           + 3 * K * 4     # read q, k, g
           + V * 4         # read v
           + 4             # read beta
           + K * V * 4     # write S_out
           + V * 4)        # write o
    return int(total_work) * per


def stats_ns(s):
    return dict(
        mean=float(statistics.fmean(s)),
        min=float(min(s)),
        max=float(max(s)),
        median=float(statistics.median(s)),
        p95=float(sorted(s)[max(0, int(0.95 * len(s)) - 1)]),
        stddev=float(statistics.pstdev(s)) if len(s) > 1 else 0.0,
    )


def bench_one(fn, B, HV, K, V, device, warmup, repeats, flush, block_cap):
    tw = B * HV
    S_in = torch.randn(B, HV, K, V, dtype=torch.float32, device=device).contiguous()
    q = torch.rand(B, HV, K, dtype=torch.float32, device=device).contiguous()
    k = torch.rand(B, HV, K, dtype=torch.float32, device=device).contiguous()
    v = torch.rand(B, HV, V, dtype=torch.float32, device=device).contiguous()
    g = torch.nn.functional.logsigmoid(
        torch.randn(B, HV, K, device=device)).contiguous()
    beta = torch.rand(B, HV, dtype=torch.float32, device=device).contiguous()
    S_out = torch.zeros(B, HV, K, V, dtype=torch.float32, device=device).contiguous()
    o = torch.zeros(B, HV, V, dtype=torch.float32, device=device).contiguous()
    torch.npu.synchronize()

    stream = torch.npu.current_stream()
    stream_ptr = getattr(stream, "_as_parameter_", None)
    bd = int(min(tw, block_cap))
    if bd < 1:
        bd = 1

    for _ in range(warmup):
        call(fn, stream_ptr, S_in, q, k, v, g, beta, S_out, o, tw, K, V, bd)
    torch.npu.synchronize()

    evs = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
           for _ in range(repeats)]
    for (e0, e1) in evs:
        flush.zero_()
        e0.record()
        call(fn, stream_ptr, S_in, q, k, v, g, beta, S_out, o, tw, K, V, bd)
        e1.record()
    torch.npu.synchronize()
    samples = [float(e0.elapsed_time(e1) * 1e6) for (e0, e1) in evs]

    st = stats_ns(samples)
    nbytes = bytes_moved(tw, K, V)
    st.update(B=B, HV=HV, K=K, V=V, units=tw, bytes_moved=nbytes,
              achieved_gbps=(nbytes / st["median"] if st["median"] > 0 else 0.0),
              block_dim=bd)
    return st


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kernel_so", nargs="?", default=_DEFAULT_SO)
    p.add_argument("--hv", type=int, default=16)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--flush-mib", type=int, default=256)
    p.add_argument("--block-cap", type=int, default=40)
    p.add_argument("--b-list", type=str, default="1,4,16,32")
    p.add_argument("--out-json", type=str, default=None)
    args = p.parse_args()

    HV, K = args.hv, args.k
    V = K
    b_values = [int(x) for x in args.b_list.split(",")]

    torch.npu.set_device(0)
    device = torch.device("npu")
    flush = torch.zeros(args.flush_mib * 1024 * 1024, dtype=torch.int8, device=device)

    fn = load_kernel(args.kernel_so)

    results = {}
    for B in b_values:
        st = bench_one(fn, B, HV, K, V, device, args.warmup, args.repeats,
                       flush, args.block_cap)
        results[str(B)] = st
        print("B=%-4d units=%-5d median=%12.1f ns  bytes=%d  BW=%.1f GB/s"
              % (B, st["units"], st["median"], st["bytes_moved"],
                 st["achieved_gbps"]))

    out = dict(stage="fused_kda_decode", timer="event",
               flush_mib=args.flush_mib, warmup=args.warmup,
               repeats=args.repeats, dims=dict(HV=HV, K=K, V=V),
               results=results)

    txt = json.dumps(out, indent=2)
    if args.out_json:
        with open(args.out_json, "w") as f:
            f.write(txt)
        print("wrote %s" % args.out_json)
    else:
        print(txt)


if __name__ == "__main__":
    main()
