"""Throughput (ms/molecule) of original vs atom_jepa_accel across batch sizes: random weights, offline.
    python benchmarks/sweep_batch_size.py
On CPUs with small caches, throughput is best at 1-2 molecules per forward; speedup of accel is similar at all sizes."""

import sys
import os
import time
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

os.environ["ATOM_JEPA_ACCEL_DISABLE"] = "1"
import atom_jepa_accel as aja
from bench_aspect import build_hub
from atom_jepa import AtomJEPA
from atom_jepa.conformers import smiles_to_samples
from mols import SMILES

samples = [smiles_to_samples(s)[0] for s in SMILES[:16]]
with tempfile.TemporaryDirectory() as hub:
    build_hub(hub)

    def load(accel):
        (aja.enable if accel else aja.disable)()
        return AtomJEPA.from_pretrained(
            os.path.join(hub, "bio"),
            device="cpu",
            precision="fp32",
            cuequivariance=False,
            verbose=False,
        )

    res = {}
    for accel in (False, True):
        m = load(accel)
        m.embed(samples[:2])
        for bs in (1, 2, 4, 8, 16):
            t = time.perf_counter()
            m.embed(samples, batch_size=bs)
            dt = time.perf_counter() - t
            res[(accel, bs)] = dt
            print(f"accel={accel!s:5} batch_size={bs:2d}  {dt / 16 * 1e3:7.0f} ms/mol", flush=True)
for bs in (1, 2, 4, 8, 16):
    print(f"bs={bs:2d} speedup {res[(False, bs)] / res[(True, bs)]:.2f}x")
