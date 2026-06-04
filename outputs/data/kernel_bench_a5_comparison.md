# A5 PTO Kernel Comparison vs A2 Saved Results

A5 PTO-only benchmark. Triton skipped. solve_tril excluded because A5 PTO tri_inverse compiles but is numerically invalid; A5 solve_tril currently uses a torch fallback for correctness only.

Shape: N_seq=16, L_seg=16384, C=128, D=128.

| H | stage | A2 ms | A5 ms | speedup | A2 est TFLOP/s | A5 est TFLOP/s |
|---:|---|---:|---:|---:|---:|---:|
| 16 | cumsum | 0.322 | 0.000 | n/a (noisy) | n/a | n/a |
| 16 | kkt | 4.668 | 5.441 | 0.86x | 29.44 | 25.26 |
| 16 | wy_fast | 6.965 | 6.770 | 1.03x | 39.46 | 40.60 |
| 16 | chunk_h | 10.121 | 12.796 | 0.79x | 27.16 | 21.48 |
| 16 | chunk_o | 11.120 | 13.919 | 0.80x | 37.08 | 29.62 |
| 32 | cumsum | 0.345 | 1.640 | n/a (noisy) | n/a | n/a |
| 32 | kkt | 9.417 | 10.911 | 0.86x | 29.19 | 25.19 |
| 32 | wy_fast | 13.090 | 13.078 | 1.00x | 42.00 | 42.04 |
| 32 | chunk_h | 20.459 | 25.934 | 0.79x | 26.87 | 21.20 |
| 32 | chunk_o | 21.959 | 27.920 | 0.79x | 37.55 | 29.54 |
| 48 | cumsum | 0.439 | 1.656 | n/a (noisy) | n/a | n/a |
| 48 | kkt | 13.680 | 16.394 | 0.83x | 30.14 | 25.15 |
| 48 | wy_fast | 20.871 | 19.669 | 1.06x | 39.51 | 41.92 |
| 48 | chunk_h | 30.183 | 39.470 | 0.76x | 27.32 | 20.89 |
| 48 | chunk_o | 33.228 | 41.754 | 0.80x | 37.23 | 29.62 |

Limitations:
- `solve_tril` is not included in the A5 PTO comparison. The A5 copy of `tri_inverse` compiles after layout fixes but produces NaNs; `solve_tril` currently uses a torch fallback only for correctness.
- `mega_kernel` has not been validated on A5 yet because it depends on the PTO triangular inverse path.
- H=64 large-shape benchmark timed out in `wy_fast`; the generated comparison includes completed H=16,32,48 rows only.
- Triton baselines were skipped because Triton is not installed in the current environment.
- `chunk_cumsum` short-run timings are noisy; the table preserves raw ms but does not treat cumsum speedup as a reliable FLOP comparison.
