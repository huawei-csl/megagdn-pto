---
name: pre-commit-validation
description: Run and resolve this repository's pre-commit lint and formatting checks before completing code changes.
---

# Pre-commit validation

Before completing code changes, run the repository's CI lint command from the repository root:

```bash
pre-commit run --all-files --show-diff-on-failure
```

The hooks check YAML, TOML, Python syntax, whitespace, merge conflicts, large files, private keys, and C/C++ formatting. If `pre-commit` is unavailable, install it with `python -m pip install pre-commit` and rerun the command.

Fix failures caused by the changes and rerun the checks. Do not claim the lint checks passed if they could not be run; report the unavailable tool or any unrelated pre-existing failures clearly.
