"""Worker: runs in a fresh subprocess, prints one ``RESULT {json}`` line."""

from __future__ import annotations

import json
import statistics
import sys
import time


def _sync(torch, device):
    if device == "cuda":
        torch.cuda.synchronize()


def main() -> int:
    cfg = json.loads(sys.argv[1])
    import torch

    torch.set_num_threads(cfg["threads"])
    import atom_jepa_accel as aja
    from atom_jepa.conformers import smiles_to_samples
    from atom_jepa.data.collate import GraphCollator
    from ._models import make_encoder
    from ._molecules import SMILES

    dev, kind = cfg["device"], cfg["kind"]
    small = kind == "grad-check"
    aja.disable()  # reference built with unpatched upstream
    ref = make_encoder(small=small).to(dev)
    new = make_encoder(small=small).to(dev)
    new.load_state_dict(ref.state_dict())
    applied = aja.accelerate(new)
    aja.enable(retrofit=False)  # retrofitting would also accelerate the reference model
    samples = [
        smiles_to_samples(s)[0]
        for s in (SMILES * (cfg["batch"] // len(SMILES) + 1))[: cfg["batch"]]
    ]
    batch = GraphCollator(cutoff=6.0)(samples)
    batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
    out = {"atoms": int(batch["atomic_numbers"].numel()), "accel_applied": bool(applied)}

    if kind == "grad-check":
        for e in (ref, new):
            e.train()
            for m in e.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.p = 0.0
            e.zero_grad()
            e(batch).pow(2).sum().backward()
        g = [
            torch.cat([p.grad.flatten() for p in e.parameters() if p.grad is not None])
            for e in (ref, new)
        ]
        err = float((g[1] - g[0]).norm() / g[0].norm())
        out.update(rel_error=err, tolerance=1e-4, passed=err < 1e-4)
        print("RESULT " + json.dumps(out))
        return 0

    def infer(e):
        e.eval()
        with torch.no_grad():
            return e(batch).detach().clone()

    def train(e):
        e.train()
        e.zero_grad()
        e(batch).float().pow(2).mean().backward()
        e.eval()

    fn = infer if kind == "inference" else train
    r0, r1 = fn(ref), fn(new)  # warm-up
    times = {"original": [], "accel": []}
    for _ in range(cfg["repeats"]):
        for label, e in (("original", ref), ("accel", new)):
            _sync(torch, dev)
            t = time.perf_counter()
            fn(e)
            _sync(torch, dev)
            times[label].append(time.perf_counter() - t)
    out.update(
        original_s=times["original"],
        accel_s=times["accel"],
        speedup=statistics.median(times["original"]) / statistics.median(times["accel"]),
    )
    if kind == "inference":
        err = float((r1 - r0).norm() / r0.norm())
        out.update(rel_error=err, tolerance=1e-5, passed=err < 1e-5)
    print("RESULT " + json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
