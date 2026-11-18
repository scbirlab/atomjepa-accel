#!/usr/bin/env python
"""Safe, local benchmark: original Atom-JEPA encoder vs atom_jepa_accel (random weights, released config).

No network, no downloads. Optional --json output only.
    python benchmarks/bench_atom_jepa.py --batch 8 --repeats 5 [--device cuda] [--compile]
Original and accelerated encoders are separate objects with identical weights; runs are interleaved
and medians reported. On CUDA the accel path is a no-op (upstream fused kernels are kept), so
speedup there should be ~1.0x -- that comparison is included deliberately.
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import torch
from mols import SMILES
import atom_jepa_accel as aja
from atom_jepa.models.jepa_equiformer import EquiformerV3Config, EquiformerV3Encoder
from atom_jepa.conformers import smiles_to_samples
from atom_jepa.data.collate import GraphCollator


def make_encoder():
    cfg = EquiformerV3Config(
        num_channels=256,
        num_layers=8,
        lmax=2,
        mmax=2,
        max_radius=6.0,
        num_radial_basis=128,
        num_heads=8,
        attn_hidden_channels=64,
        attn_alpha_channels=64,
        attn_value_channels=16,
        ffn_hidden_channels=512,
        edge_channels=128,
        attn_grid_resolution_list=[14, 8],
        ffn_grid_resolution_list=[14, 14],
        grad_checkpointing=False,
    )
    torch.manual_seed(0)
    return EquiformerV3Encoder(cfg)


def sync(d):
    if d == "cuda":
        torch.cuda.synchronize()


def timed(fn, d):
    sync(d)
    t = time.perf_counter()
    fn()
    sync(d)
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument(
        "--train-batch",
        type=int,
        default=2,
        help="training needs far more memory than inference (activations of 8 layers); keep small",
    )
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--compile",
        action="store_true",
        help="also time torch.compile'd accel encoder (CPU, slow first call)",
    )
    ap.add_argument("--json")
    a = ap.parse_args()
    d = a.device
    aja.disable()  # build the reference with unpatched upstream
    ref = make_encoder().to(d)
    new = make_encoder().to(d)
    new.load_state_dict(ref.state_dict())
    applied = aja.accelerate(new)
    batch = GraphCollator(cutoff=6.0)([smiles_to_samples(s)[0] for s in SMILES[: a.batch]])
    batch = {k: v.to(d) if torch.is_tensor(v) else v for k, v in batch.items()}
    with torch.no_grad():
        err = ((new.eval()(batch) - ref.eval()(batch)).norm() / ref(batch).norm()).item()
    print(f"accel applied: {applied}   relative error vs original: {err:.2e}")

    tbatch = GraphCollator(cutoff=6.0)([smiles_to_samples(s)[0] for s in SMILES[: a.train_batch]])
    tbatch = {k: v.to(d) if torch.is_tensor(v) else v for k, v in tbatch.items()}

    def infer(e):
        e.eval()
        with torch.no_grad():
            e(batch)

    def train(e):
        e.train()
        e.zero_grad()
        e(tbatch).float().pow(2).mean().backward()
        e.eval()

    res = {}
    for name, fn in (("inference", infer), ("train_step", train)):
        fn(ref)
        fn(new)
        r = {"original": [], "accel": []}
        for _ in range(a.repeats):
            r["original"].append(timed(lambda fn=fn: fn(ref), d))
            r["accel"].append(timed(lambda fn=fn: fn(new), d))
        mo, ma = statistics.median(r["original"]), statistics.median(r["accel"])
        res[name] = {"original_s_median": mo, "accel_s_median": ma, "speedup": mo / ma, "raw": r}
        print(
            f"{name:11s} original {mo * 1e3:8.0f} ms   accel {ma * 1e3:8.0f} ms   speedup {mo / ma:.2f}x  (median of {a.repeats})"
        )
    if a.compile and d == "cpu":
        new.body.compile_blocks(dynamic=True)
        t0 = time.perf_counter()
        infer(new)
        first = time.perf_counter() - t0
        ts = [timed(lambda: infer(new), d) for _ in range(a.repeats)]
        res["inference_compiled"] = {"first_call_s": first, "median_s": statistics.median(ts)}
        print(
            f"compiled inference: first call {first:.1f}s, steady {statistics.median(ts) * 1e3:.0f} ms"
        )
    out = {
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "threads": torch.get_num_threads(),
            "device": d,
            "gpu": torch.cuda.get_device_name() if d == "cuda" else None,
            "torch": torch.__version__,
        },
        "batch": a.batch,
        "train_batch": a.train_batch,
        "atoms": int(batch["atomic_numbers"].numel()),
        "rel_error": err,
        "results": res,
    }
    print("NOTE: results are specific to this machine, molecule set and thread count.")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
