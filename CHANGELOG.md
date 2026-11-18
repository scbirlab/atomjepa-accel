# Changelog

## Unreleased

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
