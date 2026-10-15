---
name: pto-kernel-compilation
description: Compile changed PTO kernels with the repository's Ascend build scripts and report environment limitations accurately.
---

# PTO kernel compilation

When changing a PTO kernel under `kernels/pto/`, compile the affected kernel using the script for the target architecture. Run from the repository root after sourcing the Ascend toolkit environment:

```bash
scripts/compile_kernel.sh <kernel>
```

Use `scripts/compile_a5_kernel.sh <kernel>` for A5. The kernel argument is the source basename without `.cpp`; for example, `chunk_h` compiles `kernels/pto/chunk_h.cpp`. The scripts require `ASCEND_TOOLKIT_HOME`, the Ascend `bisheng` compiler, and the matching PTO headers.

Compile all kernels when changing shared kernel interfaces, common headers, or build configuration. Check the corresponding `.github/workflows/build-single-kernels-on-a2a3.yml` or `build-single-kernels-on-a5.yml` for the complete CI kernel list. Run relevant accuracy tests when the change affects kernel behavior.

If the required Ascend compiler or toolkit is unavailable, do not claim compilation succeeded. State which check could not run and why, and rely on the matching CI build result when available.
