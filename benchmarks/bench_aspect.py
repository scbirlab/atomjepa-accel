#!/usr/bin/env python
"""Offline end-to-end benchmark through scbirlab/aspect: ("smiles", ["conformers", "atom-jepa"]).

Builds a *random-weight* checkpoint in the released on-disk format in a temp directory and points
Aspect's Hugging Face download at it, so nothing is downloaded and nothing outside the temp dir is
written. Reports conformer generation (not accelerated) separately from embedding, and interleaves
original / accelerated runs. Requires: pip install aspect-data[atom-jepa]
    python benchmarks/bench_aspect.py --n 24 --repeats 3 [--json out.json]
"""

import argparse
import dataclasses
import json
import os
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from mols import SMILES
import atom_jepa_accel as aja


def build_hub(root):
    from safetensors.torch import save_file
    from atom_jepa.models.jepa_equiformer import EquiformerV3Config, EquiformerV3Encoder

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
    enc = EquiformerV3Encoder(cfg)
    os.makedirs(os.path.join(root, "bio"))
    save_file(
        {k: v.detach().clone().contiguous() for k, v in enc.state_dict().items()},
        os.path.join(root, "bio", "model.safetensors"),
    )
    with open(os.path.join(root, "bio", "config.json"), "w") as f:
        json.dump({"eqv3_cfg": dataclasses.asdict(cfg)}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--json")
    a = ap.parse_args()
    import huggingface_hub
    import aspect.transform.atomjepa as aa
    from aspect.transform.conformers import Conformers

    with tempfile.TemporaryDirectory() as hub:
        build_hub(hub)
        huggingface_hub.hf_hub_download = lambda repo, name, **kw: os.path.join(hub, name)
        smiles = SMILES[: a.n]
        t = time.perf_counter()
        rows = Conformers()({"smiles": smiles}, "smiles")
        t_conf = time.perf_counter() - t

        def embed(accel):
            (aja.enable if accel else aja.disable)()
            aa._load_atom_jepa_cached.cache_clear()
            fn = aa.AtomJEPAEmbedding(device="cpu", batch_size=a.batch_size)
            fn({"c": rows[:1]}, "c")  # load + warm-up outside the timing
            t = time.perf_counter()
            out = fn({"c": rows}, "c")
            return time.perf_counter() - t, out

        embed(False)
        embed(True)
        res = {"original": [], "accel": []}
        for _ in range(a.repeats):
            for k, acc in (("original", False), ("accel", True)):
                dt, out = embed(acc)
                res[k].append(dt)
                res[k + "_out"] = out
        aja.enable()
    mo, ma = statistics.median(res["original"]), statistics.median(res["accel"])
    err = float(
        np.linalg.norm(res["accel_out"] - res["original_out"]) / np.linalg.norm(res["original_out"])
    )
    print(f"molecules {a.n}  threads {torch.get_num_threads()}")
    print(
        f"conformer generation (not accelerated): {t_conf:.2f} s  ({t_conf / a.n * 1e3:.0f} ms/molecule)"
    )
    print(
        f"atom-jepa embedding   original {mo:.2f} s ({mo / a.n * 1e3:.0f} ms/mol)   accel {ma:.2f} s ({ma / a.n * 1e3:.0f} ms/mol)   {mo / ma:.2f}x"
    )
    tot_o, tot_a = t_conf + mo, t_conf + ma
    print(
        f"end to end (conformers + embedding): {tot_o:.2f} s -> {tot_a:.2f} s   {tot_o / tot_a:.2f}x     relative error {err:.1e}"
    )
    if a.json:
        json.dump(
            {
                "n": a.n,
                "threads": torch.get_num_threads(),
                "t_conformers_s": t_conf,
                "embed_original_s": res["original"],
                "embed_accel_s": res["accel"],
                "rel_error": err,
            },
            open(a.json, "w"),
            indent=2,
        )


if __name__ == "__main__":
    main()
