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

- `enable()` also accelerates encoders that **already exist** (found with `gc`), so Aspect's memoized encoder is covered
  whatever the import order; `disable()` undoes it exactly, including each block's previous `optimize_grid_mlp` flag.
- When benchmarking original-versus-accelerated in one process, build the reference with the package disabled and call
  `enable(retrofit=False)`, otherwise the reference is accelerated too.
- Existing `gated_product` hooks (cuEquivariance, ROCm) are never overridden; CUDA is untouched.

## Environment switches

| Variable | Effect |
| --- | --- |
| `ATOM_JEPA_ACCEL_DISABLE=1` | don't activate on import |
