# Use with Aspect

[scbirlab/aspect](https://github.com/scbirlab/aspect) featurises tables through registered transforms; its `atom-jepa`
transform (`aspect-data[atom-jepa]`) loads Atom-JEPA weights from the Hugging Face Hub and embeds batches in-process.
atom_jepa_accel patches the upstream classes Aspect uses, so **no change to Aspect is needed**.

## Activating it

```bash
pip install atom-jepa-accel "aspect-data[atom-jepa]"
python -m atom_jepa_accel -m aspect.cli_module.cli featurize compounds.csv \
    --features "smiles:conformers:atom-jepa(batch_size=2)@aj" --output features.parquet
python -m atom_jepa_accel --also unimol_accel script.py            # both accel packages
python -m atom_jepa_accel -m pytest path/to/aspect/test        # Aspect's own tests with accel on
```

or, from Python, `import atom_jepa_accel` before the featuriser runs. An encoder that already exists when the package is imported (for example one Aspect memoized) is accelerated explicitly with `atom_jepa_accel.accelerate_existing()`; under `python -m atom_jepa_accel -m ...` the package is active before Aspect builds anything, so this is not needed.

## Input modes

Aspect's `atom-jepa` transform embeds **pre-generated conformers** (`("smiles", ["conformers", "atom-jepa"])`) by building
`{"atomic_numbers", "node_coordinates"}` samples and calling `AtomJEPA.embed(samples, layers=..., batch_size=...)`.
Passing SMILES to Aspect's transform is an error by design. The native `AtomJEPA.embed` also accepts SMILES (conformers
generated upstream); both paths end in the same encoder and agree (`tests/integration`). `layers="last"`, `"all"` and
an integer use forward hooks on blocks and are covered, as are `model="bio"` and `"molecules"`.

## Choosing `batch_size`

On this CPU throughput is best at 1–2 molecules per forward pass (large batches outgrow the cache); the accel speedup
is steady across sizes (ms per molecule through `AtomJEPA.embed`, 16 molecules, 2-vCPU VM):

| `batch_size` | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| original | 830 | 812 | 1034 | 1206 | 1248 |
| with atom-jepa-accel | 365 | 364 | 424 | 515 | 543 |

Aspect passes `batch_size` straight through, e.g. `atom-jepa(batch_size=2)`. The optimum depends on the CPU's caches
and core count, so measure with `python benchmarks/sweep_batch_size.py` rather than copying this value. Conformer
generation (≈30 ms/molecule) is negligible next to embedding.

## What was and was not verified

- All integration tests (`tests/integration/`) run the real Aspect code with **random weights** saved in the real
  on-disk format, with only the Hugging Face download redirected to a temporary directory. Pretrained weights could
  not be downloaded in the environment where this was developed; the arithmetic is the same, but please also run
  Aspect's own tests on a machine where the weights are cached.
- An end-to-end `aspect featurize` CLI run (CSV in, Parquet out) with and without the runner agreed: Atom-JEPA features within 1.4e-7 relative error on a 12-molecule table.
- Aspect keys its feature cache on its own settings; accelerated outputs differ from the original at fp32 rounding level (≈1e-7 relative), **not bit-identical**; features cached without accel remain valid for any downstream use but would not match byte-for-byte.
