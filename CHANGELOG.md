# Changelog

## Unreleased

- Verified against `atom-jepa` 0.4.0 (PyPI) and the repository at commit `64d539b` (identical hashes).
- `python -m atom_jepa_accel ...` runner to activate the package for programs that cannot be edited (for example the
  `scbirlab/aspect` CLI), with `--also` to combine both accel packages.
- Aspect integration tests (`tests/integration/`) with random-weight checkpoints and a mocked Hugging Face hub.
- `python -m atom_jepa_accel.bench`: offline benchmark runner (fresh subprocess per configuration, time budget, equivalence
  gate) and a self-contained HTML report.
- Documentation site (Sphinx), CI matrix, CONTRIBUTING, AGENTS.md.

## 0.1.0

- Initial release.
