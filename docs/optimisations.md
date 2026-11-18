# How it works

## What the profile showed

The encoder (EquiformerV3: SO(3) Wigner-D rotations, SO(2) linears, gated activations on an S2 grid, per-edge messages)
spends a large share of CPU time in the **grid round trip** of the gated activations
(`to_grid → multiply → from_grid`) and in memory-bound per-edge tensors, not in the big GEMMs. Upstream already has
fused hooks for CUDA (`fused_gather_rotate`, `gated_product`, …) and enables `optimize_grid_mlp` only there.

## The two changes

1. **Enable the grid-MLP path on CPU** (`AtomJEPA._configure_execution` / `set_grid_mlp_optimization`): ≈1.40× on its own.
2. **Exact trilinear coupling.** With `T` = `to_grid_mat` and `P` = `from_grid_mat`,

   `from_grid(to_grid(a) ⊙ to_grid(b)) = C · (a ⊗ b)`, `C[o,i,j] = Σ_g P[o,g] T[g,i] T[g,j]`,

   built once in float64 per module and cached per dtype/device. `C` has 83 structural non-zeros of 729 (the Gaunt
   selection rules; 84–89 entries exceed 1e-7 because the grid matrices are stored in float32), and a dense GEMM over
   the flattened outer product replaces two grid transforms. Relative error versus the original is ≈1.4e-7 (fp32
   rounding): **not bit-identical**. The tests cover lmax 1–3, 1–256 channels, activation scales 1e-3…1e3 (float32 within
   2e-5, float64 within 1e-12), gradients, 25-step Adam trajectories and the production-size encoder.

**Relationship to upstream and to Gaunt/Clebsch–Gordan products.** This coupling is not new: upstream's cuEquivariance
and ROCm gated products build the same tensor from the same grid matrices, and the grid product of two S2 functions
followed by projection is the Gaunt tensor product. What this package adds is the CPU realisation and enabling it.

### Dense GEMM versus alternatives (CPU)

`experiments/coupling_alternatives/bench_coupling.py`, 256 nodes × 256 channels, 2 threads, medians of interleaved runs,
first variant = upstream's grid route:

| variant | ms | vs grid route |
|---|---:|---:|
| grid route (`to_grid → ⊙ → from_grid`) | 3.53 | 1.00× |
| **dense GEMM (shipped)** | **2.37** | **1.49×** |
| indexed contraction over the 83 non-zeros (gather, multiply, `index_add_`) | 7.82 | 0.45× |
| per-output small GEMMs over the non-zero (i, j) pairs | 7.53 | 0.47× |
| e3nn channelwise Clebsch–Gordan `TensorProduct` (l ≤ 2, unit weights; *different numbers*, timing only) | 7.86 | 0.45× |

On this 2-vCPU CPU the sparsity does not pay: gathers and small kernels lose to one large MKL GEMM. A many-core CPU or a
fused sparse kernel could change that; this is a measurement on one machine, not a general claim. The 2.6× whole-encoder
speed-up also includes the grid-MLP enable (≈1.4× alone), not just this operator.

## Results (2-vCPU Intel Xeon @ 2.3 GHz VM, PyTorch 2.15, random weights at the released config: 256 ch, 8 layers, lmax 2)

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

The speedup is stable at ≈2.2–2.4× across batch sizes. Separately, throughput on this CPU is best at 1–2 molecules per
forward pass (≈1.5× better than batch 16 for both), a machine-dependent effect; use `benchmarks/sweep_batch_size.py`.
End to end through Aspect with 24 molecules in one batch: 1654 → 878 ms/molecule (**1.88×**). Training at ≥ 4 molecules
was OOM-killed on the 7 GB test VM (with the unpatched model), so training numbers use 2.

## Opt-in extras (not enabled)

- `body.compile_blocks(dynamic=True)` on CPU: a one-off test gave a steady 1.63 s vs ≈2.2 s at batch 8 but with a
  **15.9 s first call**, outside the interleaved protocol. Indicative only.
- bf16 autocast on AMX CPUs: ≈2e-3 relative error; not validated, not default.

## Negative results (kept as evidence)

- **`packed_gemm`** (SO(2) packed GEMM toggle) did not help on this CPU.
- **A C++ gather/rotate kernel was not written.** After the above, time is dominated by edge-wise GEMMs and memory-bound
  gather/scatter; there was no measurement suggesting a hand kernel would beat MKL on two cores.
- Conformer generation (RDKit) is a separate, unaccelerated fixed cost (≈30 ms/molecule, negligible here).

## Not measured

Many-core CPUs, CUDA (the accel path is a no-op there), and real pretrained weights (Hugging Face was not reachable from
the development sandbox).
