# Installation

```bash
pip install torch atom-jepa>=0.4     # your usual way; CPU or CUDA build both fine
pip install atom-jepa-accel                          # pure-Python wheel; no compiler needed
```

For development: `pip install -e ".[test]"` from a checkout (see {doc}`contributing`).

Importing the package never compiles anything, never touches the network and never fails: if the upstream version is
not one of the verified ones, the original code runs and `atom_jepa_accel.status()` says why.

| | Supported / tested |
| --- | --- |
| Atom-JEPA | `atom-jepa` 0.4.0 (PyPI) and the repository at commit `64d539b` (identical hashes) |
| Python | >= 3.10; developed and measured on 3.13; CI matrix 3.10–3.14 plus non-blocking 3.15 (workflow written but not yet run) |
| PyTorch | developed and measured with 2.15 (CPU); older versions are untested |
| GPU | not accelerated; see {doc}`optimisations` |
