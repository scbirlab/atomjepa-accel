# Changelog

## Unreleased

- **Fixed:** accelerated encoders with grid dropout crashed in training ("grid fusion requires inactive grid dropout").
  A per-block forward pre-hook now switches the coupling off for such layers while training and on in eval mode, so models
  configured with dropout are still accelerated for inference. Seeded dropout training matches the original (new tests:
  zero dropout, dropout in train, dropout in eval, `Dropout(p=0)`).
- **Documented:** the pre-hook is mutable execution state. Sequential forwards are supported; same-mode concurrent forwards are unvalidated; mixed-mode concurrency and mode changes during a forward are unsupported.
- **Tests:** whole-encoder numerical tests now build the reference with the package disabled, assert it carries no hooks
  before and after accelerating the second encoder, so they cannot compare two accelerated models.
- **Changed:** retrofit of already-built encoders is now opt-in (`accelerate_existing()`, `enable(retrofit=True)` or
  `ATOM_JEPA_ACCEL_RETROFIT=1`) instead of a default `gc.get_objects()` scan.
- **Changed:** `GatedGridProduct` caches its float64 master coupling per dtype/device, rebuilds it when the grid matrices
  change, uses a float64 coupling for float64 inputs, and validates input shapes.
- Numerical tests beyond the smoke configuration (`tests/test_numerics.py`): lmax 1–3, 1–256 channels, scales 1e-3…1e3,
  float64, gradients, 25-step trajectories, stale-cache cases, and a `slow` production-size test.
- `experiments/coupling_alternatives/`: dense GEMM vs indexed contraction vs e3nn Clebsch–Gordan on CPU.
- Docs state plainly that the coupling is upstream's own and the Gaunt tensor product; the contribution is the CPU path.
- Fixed: benchmark jobs in `docs.yml` failed with "SMILES input needs RDKit". New `[bench]` extra (RDKit), installed by the
  workflow; the benchmark now checks for RDKit up front and exits with an install hint; import-time `FutureWarning` noise is
  hidden in worker processes.
- README: badges and links to the docs, benchmark methodology and live report.
- Verified against `atom-jepa` 0.4.0 (PyPI) and the repository at commit `64d539b` (identical hashes).
- `python -m atom_jepa_accel ...` runner to activate the package for programs that cannot be edited (for example the
  `scbirlab/aspect` CLI), with `--also` to combine both accel packages.
- Aspect integration tests (`tests/integration/`) with random-weight checkpoints and a mocked Hugging Face hub.
- `python -m atom_jepa_accel.bench`: offline benchmark runner (fresh subprocess per configuration, time budget, equivalence
  gate) and a self-contained HTML report.
- Documentation site (Sphinx), CI matrix, CONTRIBUTING, AGENTS.md.

## 0.1.0

- Initial release.
