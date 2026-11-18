# Compatibility and safety

**Verified against:** `atom-jepa` 0.4.0 (PyPI) and the repository at commit `64d539b` (identical hashes).

Each replaced function is identified by a SHA-256 prefix of its exact source. If the installed Atom-JEPA differs, that patch
is **not applied** and the original code runs; `atom_jepa_accel.status()` reports the unverified hash. This hash list is the
safety mechanism: do not add a hash without reading the upstream diff and re-verifying the replacement (see
{doc}`contributing`).

| Guarantee | |
| --- | --- |
| Parameters, buffers, `state_dict` keys | unchanged; checkpoints are interchangeable with and without the package |
| Importing | no network, no compilation, no change to threads, seeds or dtype |
| `disable()` | restores the original function objects (and, for Atom-JEPA, every encoder it touched) |
| Unknown upstream source | original code runs |
| Network | none used by the package, its tests or its benchmarks |

## Behaviours worth knowing

- Encoders that **already exist** at import time are left alone. `accelerate(model)` handles one model;
  `accelerate_existing()` or `ATOM_JEPA_ACCEL_RETROFIT=1` handle all of them (a `gc.get_objects()` scan, hence opt-in).
  `disable()` undoes either exactly, including each block's previous `optimize_grid_mlp` flag.
- When benchmarking original-versus-accelerated in one process, keep retrofit off (the default), otherwise the
  reference is accelerated too.
- **Grid dropout.** Upstream's `gated_product` hook raises in training when grid dropout is active (FFN grid MLP with
  `dropout > 0`; attention activation with any non-`Identity` `grid_drop`). A forward pre-hook on each block turns the
  coupling off for such layers while training (upstream's own CPU-enabled path runs) and on again in eval mode.
  Seeded training with dropout matches the original (tested).
- **Thread safety.** The per-block pre-hook (`_toggle`) assigns `owner.gated_product` on every forward, so a model carries
  mutable execution state. Sequential forwards are supported. Concurrent forwards in the same mode avoid the known
  train/eval switching hazard but have not been independently validated. Concurrent forwards in different modes, or changes
  to `train()`/`eval()` during a forward, are unsupported (a forward could see the other mode's hook, and upstream raises if
  the hook is live while training with grid dropout). Use one model object per mode or per thread. For a model without grid
  dropout, `_toggle()` does not change `gated_product` after installation, assuming nothing else modifies it; that is a
  statement about this mechanism, not a general thread-safety guarantee for the model.
- **Stale state.** The coupling is cached per dtype/device and rebuilt when the grid matrices are replaced or edited in
  place; `model.double()` / `.to(device)` therefore cannot leave it stale, and float64 inputs use a float64 coupling.
- Existing `gated_product` hooks (cuEquivariance, ROCm) are never overridden; CUDA is untouched.

## Environment switches

| Variable | Effect |
| --- | --- |
| `ATOM_JEPA_ACCEL_DISABLE=1` | don't activate on import |
| `ATOM_JEPA_ACCEL_RETROFIT=1` | on activation, also accelerate encoders that already exist |
