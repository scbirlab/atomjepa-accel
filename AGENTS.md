# AGENTS.md — instructions for AI coding agents

Project: **atom-jepa-accel** — `import atom_jepa_accel` makes the Atom-JEPA (EquiformerV3) encoder faster on CPU by enabling the grid-MLP path (upstream only does so on CUDA) and replacing the S2-grid round trip in the gated activations by an exactly equivalent sparse trilinear coupling. Pure PyTorch. Human-oriented docs are in `README.md` and `CONTRIBUTING.md`; this file is the
machine-oriented summary. Keep it in sync with the code.

## Commands (run from the repo root)

```bash
pip install -e ".[test]"                    # plus the upstream package and torch
pytest -q                                   # unit tests (tests/integration needs: pip install "aspect-data[atom-jepa]")
ruff check . && ruff format --check .
sphinx-build -W -b html docs docs/_build/html
python -m atom_jepa_accel.bench --preset quick --output /tmp/b        # end-to-end speed + equivalence (offline)
python -m atom_jepa_accel status                      # what is active
```

Prerequisites: Python >= 3.10, `torch>=2.1`, `atom-jepa`. Verified against `atom-jepa` 0.4.0 (PyPI) and the repository at commit `64d539b` (identical hashes).

## Architecture in one paragraph

`enable()` replaces two upstream methods, each only if its source hash is in `KNOWN_HASHES`:
`HeadlessEquiformerV3.set_grid_mlp_optimization` (after the original runs, on CPU, call `accelerate`) and
`AtomJEPA._configure_execution` (on CPU, call `set_grid_mlp_optimization(True)`, which upstream skips off CUDA).
`accelerate(model_or_body)` sets `gated_product = GatedGridProduct(grid)` on each block's attention activation and FFN
grid MLP (only if none is set, so cuEquivariance/ROCm hooks are never overridden) and turns on `optimize_grid_mlp`,
recording the previous flag so `unaccelerate` restores it. `GatedGridProduct` is a plain callable (not an
`nn.Module`) holding the float64-derived coupling matrix, so `state_dict` is unchanged. `enable(retrofit=True)` also
accelerates encoders that already exist (found via `gc`), tracked in a `WeakSet` so `disable()` can undo it.

## Invariants — do not break these

- `from_grid(to_grid(a) * to_grid(b)) * gate == C . (a (x) b) * gate` with `C[o,i,j] = sum_g P[o,g] T[g,i] T[g,j]`;
  `C` is derived in float64 from each module's own grid matrices. Results agree with the original to fp32 rounding
  (about 1e-7 relative); they are **not** bit-identical, and docs must say so.
- No parameters, buffers or `state_dict` keys change; the operator is not a registered module.
- CUDA is untouched (upstream fused kernels stay); existing `gated_product` hooks are never overridden.
- Unknown upstream source hash => the **original** code runs and `status()` says why. Automatic paths never raise
  into user code (`_auto_accelerate`).
- `disable()` restores methods and every touched encoder exactly, including the previous `optimize_grid_mlp` flag.

## Things agents commonly get wrong here

- **A/B comparisons**: `enable()` retrofits every existing encoder, including your reference model. Build the reference
  with the package disabled and call `enable(retrofit=False)` afterwards (see `bench/_worker.py`), otherwise you will
  time accelerated code against itself (speedup ~1.0, error exactly 0).
- Don't add a hash to `KNOWN_HASHES` without reading the upstream diff and re-verifying.
- Don't claim a speedup without `python -m atom_jepa_accel.bench` (or `benchmarks/`) numbers on a stated CPU. Never
  fabricate or extrapolate timings (all published numbers: 2-vCPU VM; training at >= 4 molecules ran out of memory
  on a 7 GB machine).
- Throughput on small-cache CPUs is best at 1-2 molecules per forward; report speedups at equal batch size and also
  best-vs-best.
- Real pretrained weights were not available when this was written; tests use random weights in the released
  on-disk layout. Keep CI free of downloads.
- Don't run `pytest` while timing.
- Negative results belong in `docs/optimisations.md` (don't delete them).
- Read-only default: don't push, tag, publish or write to any remote service without the maintainer's instruction.

## Style

Ruff (`line-length = 100`), type hints on public API, NumPy-style docstrings, English (British spellings fine).
Commit messages: short imperative subject (`perf: ...`, `fix: ...`).
