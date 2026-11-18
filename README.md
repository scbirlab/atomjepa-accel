# atom-jepa-accel

<p align="left">
<a href="https://github.com/scbirlab/atomjepa-accel/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/scbirlab/atomjepa-accel/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://scbirlab.github.io/atomjepa-accel/"><img alt="Docs" src="https://img.shields.io/badge/docs-GitHub%20Pages-blue"></a>
<a href="https://scbirlab.github.io/atomjepa-accel/benchmarks/report/"><img alt="Benchmarks" src="https://img.shields.io/badge/benchmarks-live%20report-informational"></a>
<img alt="Python" src="https://img.shields.io/badge/python-3.11%E2%80%933.14-blue">
<img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>


Drop-in **CPU** acceleration for [Atom-JEPA](https://github.com/khelverskovp/atom-jepa) (EquiformerV3 encoder), activated by one import. Pure PyTorch; parameters, buffers and `state_dict` keys are unchanged. Works under [scbirlab/aspect](https://github.com/scbirlab/aspect) (`atom-jepa` featuriser) without editing Aspect.

```python
import atom_jepa_accel  # patches applied on import; models built before OR after are accelerated
import atom_jepa  # ordinary code, unchanged

atom_jepa_accel.status()
atom_jepa_accel.accelerate(model)
atom_jepa_accel.unaccelerate(model)
atom_jepa_accel.disable()
```
`ATOM_JEPA_ACCEL_DISABLE=1` prevents automatic activation. Encoders that were built *before* the import are only accelerated on request: `atom_jepa_accel.accelerate(model)` for one model, `atom_jepa_accel.accelerate_existing()` for all of them (scans `gc.get_objects()`), or `ATOM_JEPA_ACCEL_RETROFIT=1`.

## Use with Aspect (or any program you don't want to edit)

```bash
pip install atom-jepa-accel "aspect-data[atom-jepa]"
python -m atom_jepa_accel -m aspect.cli_module.cli featurize compounds.csv \
    --features "smiles:conformers:atom-jepa(batch_size=2)@aj" --output features.parquet
python -m atom_jepa_accel --also unimol_accel script.py          # activate both accel packages
python -m atom_jepa_accel -m pytest path/to/aspect/test          # run Aspect's own tests with accel on
```
Aspect only embeds pre-generated conformers (`("smiles", ["conformers", "atom-jepa"])`); the native `AtomJEPA.embed` also accepts SMILES (conformers generated upstream). Both end in the same encoder and are covered by `tests/integration/`, including `layers="last"|"all"|<int>` (forward hooks on blocks), both released models, several conformers per molecule spanning batches, and an encoder that Aspect already loaded and memoized before this package was imported (accelerated explicitly with `accelerate_existing()`; `disable()` restores it exactly). Under `python -m atom_jepa_accel -m aspect...` the package is active before Aspect builds anything, so no retrofit is needed.

## What it does

1. **Enables the grid-MLP path on CPU.** Upstream turns on `optimize_grid_mlp` only for CUDA; on CPU it was left off (≈1.4× on its own).
2. **Exact sparse trilinear coupling for gated activations.** `from_grid(to_grid(a)·to_grid(b))·gate` equals a fixed tensor `C[o,i,j] = Σ_g P[o,g]T[g,i]T[g,j]` contracted with the outer product of the two coefficient halves; `C` is built once in float64 from each module's own grid matrices (≈85 non-zeros of 729). Algebraically exact; measured relative error vs original ≈ 1.4e-7 (fp32 rounding, **not bit-identical**, so cached features computed with and without accel differ at that level).

It never overrides an existing `gated_product` hook (cuEquivariance/ROCm) and does nothing on CUDA. **Verified against:** `atom-jepa` 0.4.0 (PyPI) and repo commit `64d539b` (identical hashes). Anything else keeps the original code and `status()` says why.

## Measured results

Hardware: **2 vCPU cloud VM (Intel Xeon @ 2.3 GHz), 7 GB RAM**, PyTorch 2.15.0, random weights at the released config (256 channels, 8 layers, lmax 2), interleaved runs, medians. Not generalisable.

| | original | atom-jepa-accel | speedup |
|---|---|---|---|
| encoder inference, 8 molecules | 5812 ms | 2224 ms | **2.61×** |
| training step, 2 molecules | 4005 ms | 1692 ms | **2.37×** |

Through `AtomJEPA.embed` (what Aspect calls), 16 molecules, ms per molecule:

| batch size | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| original | 830 | 812 | 1034 | 1206 | 1248 |
| atom-jepa-accel | 365 | 364 | 424 | 515 | 543 |
| speedup | 2.27× | 2.23× | 2.44× | 2.34× | 2.30× |

Speedup is stable at ≈2.2–2.4× across batch sizes. **Separately, on this CPU throughput is best at 1–2 molecules per forward** (large batches outgrow the cache), worth another ≈1.5× relative to batch 16 for both original and accel. Aspect passes `batch_size` straight through (`atom-jepa(batch_size=2)`), so that needs no code change; the optimum is machine-dependent, so measure with `benchmarks/sweep_batch_size.py` (many-core machines may prefer larger batches). End to end through Aspect (24 molecules in one batch of 32): 1654 → 878 ms/molecule (**1.88×**). Conformer generation (≈30 ms/molecule) is negligible next to embedding.

Training at ≥ 4 molecules was OOM-killed on the 7 GB test machine (with the unpatched model), so training numbers use 2 molecules.

Optional, off by default: `body.compile_blocks(dynamic=True)` on CPU gave a steady 1.63 s vs ≈2.2 s at batch 8 in a one-off test with a **15.9 s first call**, not under the interleaved protocol — indicative only. bf16 on AMX CPUs was ≈2e-3 relative error; not validated.

## What is and is not new here

The coupling `C[o,i,j] = Σ_g P[o,g] T[g,i] T[g,j]` is **upstream's own**: the cuEquivariance (`admet_cue_ops.GatedGridProduct`) and ROCm (`rocm_grid.ROCmGridProduct`) hooks build exactly this tensor from the model's grid matrices. It is also the Gaunt tensor product of the two inputs (the pointwise product of two S2 functions, projected back), a known construction in equivariant networks. This package contributes the CPU realisation (one dense GEMM; enabling the grid-MLP path that upstream turns on only for CUDA), the safety rails (hash guards, reversibility, hook and dropout handling) and the measurements, not new mathematics. `experiments/coupling_alternatives/` compares the dense GEMM with an indexed contraction over its 83 structural non-zeros and with an e3nn Clebsch-Gordan tensor product on the same shapes; on this CPU the dense GEMM is the fastest (see [optimisations](docs/optimisations.md)).

## Negative results

- `packed_gemm` did not help on this CPU.
- A C++ gather/rotate kernel was **not written**: after the above, time is dominated by edge-wise GEMMs and memory-bound gather/scatter, and I had no measurement suggesting a hand kernel would beat MKL on 2 cores.

## Limits of what was verified

- **Real pretrained weights could not be downloaded** (Hugging Face blocked), so equivalence tests use random weights saved in the real released layout and exercise the real `AtomJEPA.from_pretrained`/`embed` and Aspect code. The maths is exact, so real weights are expected to behave the same; please run `python -m atom_jepa_accel -m pytest path/to/aspect/test` where the weights are cached.
- Many-core CPUs and CUDA were not measured.
- **Thread safety.** The per-block pre-hook updates `owner.gated_product` on each forward (mutable state). Sequential forwards are supported; concurrent forwards in the same mode avoid the known train/eval switching hazard but have not been independently validated; concurrent forwards in different modes, or `train()`/`eval()` changes during a forward, are unsupported (use one model object per mode or thread). See [compatibility](docs/compatibility.md).
- **Grid dropout.** Upstream's `gated_product` hook raises in training when a layer has grid dropout (the FFN grid MLP with `dropout > 0`; the attention activation with any non-`Identity` `grid_drop`, even `Dropout(p=0)`), which made accelerated models with dropout crash in training. A forward pre-hook on each block now switches those layers to upstream's own CPU-enabled path while training and back to the coupling in eval mode, so such a model is accelerated for inference and trains correctly (tested: zero dropout, dropout in training, dropout in eval, `Dropout(p=0)`, seeded equality with the original).

## Tests and benchmarks

```bash
pip install -e .                   # plus: pip install "aspect-data[atom-jepa]" for tests/integration
pytest tests                       # forward, gradients, state_dict, reversibility, opt-in retrofit, hook safety, grid dropout,
                                   # numerics (lmax 1-3, 1-256 channels, 1e-3..1e3 scales, float64, 25-step training), Aspect
pytest -m slow tests/test_numerics.py   # released production-size configuration (256 ch, 8 layers)
python benchmarks/bench_atom_jepa.py --batch 8 --train-batch 2 --repeats 5 [--device cuda] [--compile] [--json out.json]
python benchmarks/sweep_batch_size.py
python benchmarks/bench_aspect.py --n 24 --repeats 3
```
Benchmarks use random weights, make no network access and write only the optional JSON.

## Benchmark it yourself, offline

```bash
python -m atom_jepa_accel.bench               # quick preset, ~2-5 min; writes ./atom_jepa_accel_bench/report-<host>.html
python -m atom_jepa_accel.bench --dry-run     # show exactly what would run
python -m atom_jepa_accel.bench --preset standard [--device cuda]
```
Each configuration runs in a fresh subprocess with a timeout and a global time budget; random weights, no network, writes only into
`--output`; original and accelerated code are timed interleaved, and the exit status is non-zero if outputs or gradients disagree.

## Documentation, contributing, CI

**Docs:** <https://scbirlab.github.io/atomjepa-accel/> · [Optimisations](https://scbirlab.github.io/atomjepa-accel/optimisations.html) · [Aspect usage](https://scbirlab.github.io/atomjepa-accel/aspect.html) · [Compatibility](https://scbirlab.github.io/atomjepa-accel/compatibility.html) ·
**Benchmarks:** [methodology and results](https://scbirlab.github.io/atomjepa-accel/benchmarks/index.html) · [live multi-runner report (CI, latest main)](https://scbirlab.github.io/atomjepa-accel/benchmarks/report/)
(these URLs work once the owner enables GitHub Pages and `docs.yml` has run).

- Docs (Sphinx, pydata theme, modelled on [scbirlab/chemprop-accel](https://github.com/scbirlab/chemprop-accel)): `docs/`;
  build with `pip install -e ".[docs]" && sphinx-build -W -b html docs docs/_build/html`. It builds warning-free here; publishing to
  GitHub Pages is done by `.github/workflows/docs.yml` once the owner enables Pages.
- [CONTRIBUTING.md](CONTRIBUTING.md), [AGENTS.md](AGENTS.md), [CHANGELOG.md](CHANGELOG.md), [SECURITY.md](SECURITY.md).
- CI (`.github/workflows/ci.yml`, `docs.yml`): lint, Python 3.11-3.14 matrix (+ non-blocking 3.15), an upstream-`main` canary, Aspect integration, macOS/Windows/arm,
  packaging, and a multi-runner benchmark report. **These workflows have been linted with actionlint but not run**: this repository was
  prepared read-only and nothing has been pushed. Locally everything was run on Python 3.13 only.
