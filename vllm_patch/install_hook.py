#!/usr/bin/env python3
"""Apply in-source edits to the installed vllm-ascend package for PTO hook support.

The script is **idempotent**. After running once, set ``VLLM_PTO_PATCH_DIR`` at
runtime to activate the PTO patch.

**v0.18 (and similar layouts):** patches three files under ``vllm_ascend``:

1. ``patch/worker/__init__.py`` — injects an early hook that calls
   ``apply_pto_patch()`` when ``VLLM_PTO_PATCH_DIR`` is set, after Triton
   patches load and before Qwen worker modules import ``chunk_gated_delta_rule``.

2. ``patch/worker/patch_qwen3_5.py`` — switches from a static import to a
   dynamic ``fla_ops`` lookup so monkey-patches take effect at call time.

3. ``patch/worker/patch_qwen3_next.py`` — same for the MoE / Next path.

**v0.19+:** the worker hook is unchanged. Qwen patch files may be missing or no
longer import ``chunk_gated_delta_rule`` (GDN prefill uses ``vllm_ascend.ops.gdn``);
``install_hook.py`` skips those edits. Runtime routing is handled in
``apply.py`` (patches ``vllm.model_executor.layers.fla.ops`` and
``vllm_ascend.ops.triton.fla.chunk``).

Usage::

    python vllm_patch/install_hook.py
    python vllm_patch/install_hook.py --dry-run
    python vllm_patch/install_hook.py --vllm-ascend-root /path/to/vllm_ascend
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HOOK = '''
    # PTO kernel swap: MUST run before patch_qwen3_next / patch_qwen3_5 import
    # chunk_gated_delta_rule (those modules cache the reference at import time).
    try:
        import os as _pto_os
        import sys as _pto_sys

        _pto_dir = _pto_os.environ.get("VLLM_PTO_PATCH_DIR")
        if _pto_dir and _pto_os.path.isdir(_pto_dir):
            if _pto_dir not in _pto_sys.path:
                _pto_sys.path.insert(0, _pto_dir)
            from apply import apply_pto_patch  # type: ignore  # noqa: E402

            apply_pto_patch()
    except Exception as _pto_exc:
        import warnings as _pto_warnings

        _pto_warnings.warn(f"VLLM_PTO_PATCH_DIR apply_pto_patch failed: {_pto_exc!r}", stacklevel=1)

'''

_SENTINEL = '_pto_dir = _pto_os.environ.get("VLLM_PTO_PATCH_DIR")'


def _default_root() -> Path:
    import vllm_ascend
    return Path(vllm_ascend.__file__).resolve().parent


def _hook_in_place(text: str) -> bool:
    if _SENTINEL not in text:
        return False
    i_hook = text.find(_SENTINEL)
    i_anchor = text.find("import vllm_ascend.patch.worker.patch_weight_utils")
    return i_anchor == -1 or i_hook < i_anchor


def _remove_old_trailing_hook(text: str) -> str:
    mark = "# Optional out-of-tree PTO swap for ``chunk_gated_delta_rule``"
    if mark not in text:
        return text
    idx = text.find(mark)
    anchor = text.find("import vllm_ascend.patch.worker.patch_weight_utils")
    if anchor != -1 and idx < anchor:
        return text
    if "apply_pto_patch()" not in text[idx:]:
        return text
    return text[:idx].rstrip() + "\n"


def _insert_worker_hook(text: str) -> str:
    anchor = "    import vllm_ascend.patch.worker.patch_v2.patch_triton  # noqa\n"
    i = text.find(anchor)
    if i == -1:
        raise RuntimeError("Anchor 'patch_v2.patch_triton' not found in worker/__init__.py")
    j = i + len(anchor)
    while j < len(text) and text[j] == "\n":
        j += 1
    insert_at = text.find("# isort: off", j)
    if insert_at == -1:
        raise RuntimeError("'# isort: off' not found after anchor in worker/__init__.py")
    return text[:insert_at] + _HOOK + "\n" + text[insert_at:]


def _patch_qwen_file(text: str, *, path: Path) -> str | None:
    if "_vllm_fla_ops.chunk_gated_delta_rule" in text:
        return None  # already patched

    if "patch_qwen3_5" in path.name:
        old = "from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule"
        new = (
            "import vllm.model_executor.layers.fla.ops as _vllm_fla_ops\n"
            "from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule"
        )
        if old not in text:
            raise RuntimeError(f"{path}: expected import not found; patch manually.")
        text = text.replace(old, new, 1)
    else:
        old = "from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule\n"
        new = "import vllm.model_executor.layers.fla.ops as _vllm_fla_ops\n"
        if old not in text:
            raise RuntimeError(f"{path}: expected import not found; patch manually.")
        text = text.replace(old, new, 1)

    text = text.replace(
        ") = chunk_gated_delta_rule(\n",
        ") = _vllm_fla_ops.chunk_gated_delta_rule(\n",
        1,
    )
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vllm-ascend-root", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-worker-hook", action="store_true")
    ap.add_argument("--skip-qwen-patch", action="store_true")
    args = ap.parse_args()

    root = args.vllm_ascend_root or _default_root()

    # 1. Worker hook
    if not args.skip_worker_hook:
        target = root / "patch" / "worker" / "__init__.py"
        if not target.is_file():
            print(f"ERROR: {target} not found", file=sys.stderr)
            return 2
        text = target.read_text("utf-8")
        text = _remove_old_trailing_hook(text)
        if _hook_in_place(text):
            print(f"OK (already applied): {target}")
        else:
            new_text = _insert_worker_hook(text)
            if args.dry_run:
                print(f"DRY-RUN: would write worker hook → {target}")
            else:
                target.write_text(new_text, "utf-8")
                print(f"OK: worker hook written → {target}")

    # 2. Qwen model patches (v0.18: static ``chunk_gated_delta_rule`` import in these files;
    # v0.19+: GDN uses ``vllm_ascend.ops.gdn`` + ``apply_pto_patch`` on the Ascend chunk module.)
    if not args.skip_qwen_patch:
        for name in ("patch_qwen3_5.py", "patch_qwen3_next.py"):
            p = root / "patch" / "worker" / name
            if not p.is_file():
                print(f"SKIP (not present): {p}")
                continue
            try:
                new_text = _patch_qwen_file(p.read_text("utf-8"), path=p)
            except RuntimeError as exc:
                if "expected import not found" in str(exc):
                    print(f"SKIP (layout differs): {p} — {exc}")
                    continue
                raise
            if new_text is None:
                print(f"OK (already applied): {p}")
            elif args.dry_run:
                print(f"DRY-RUN: would patch {p}")
            else:
                p.write_text(new_text, "utf-8")
                print(f"OK: patched {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
