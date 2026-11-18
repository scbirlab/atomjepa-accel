## What and why

## Checklist
- [ ] `ruff check . && ruff format --check .` pass
- [ ] `pytest` passes locally (and `tests/integration` if `aspect-data[atom-jepa]` is installed)
- [ ] New/changed behaviour has tests (equivalence vs the original Atom-JEPA code where relevant)
- [ ] Performance claims are backed by `python -m atom_jepa_accel.bench` output (attach JSON / table, state the CPU)
- [ ] Docs / CHANGELOG updated
- [ ] If an upstream function hash changed: `KNOWN_HASHES` updated **and** the replacement re-verified
