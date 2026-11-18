"""CPU timing of ways to apply the gated S2-grid product  y = from_grid( to_grid(a) * to_grid(b) ) * gate.

    python experiments/coupling_alternatives/bench_coupling.py [--nodes 512] [--channels 256] [--repeats 7]

Variants (all give the same function except where noted; each is checked against the first):
  grid      upstream's pointwise route: to_grid -> multiply -> from_grid (what runs without the package)
  dense     atom_jepa_accel.GatedGridProduct: one dense GEMM over the flattened outer product (shipped)
  indexed   the same coupling restricted to its ~83 structural non-zeros: gather, multiply, index_add
  per-output  dense, but only the structurally non-zero (i, j) pairs of each output block as separate small GEMMs
  e3nn-cg   e3nn channelwise TensorProduct over the same l <= 2 irreps (Clebsch-Gordan). NOT the same numbers: CG paths
            carry different normalisation than the Gaunt coupling, so it is timed for cost only, with unit weights.
Random inputs, no weights, offline. Medians of interleaved repeats, so shared-VM noise is visible in the spread.
"""

import argparse
import statistics
import time
import warnings

warnings.filterwarnings("ignore")
import torch

import atom_jepa_accel as aja
from atom_jepa.models.equiformer_v3.so3 import SO3Grid


def indexed_factory(prod):
    C = prod.coupling(torch.float32, "cpu").reshape(9, 9, 9)
    o, i, j = (C.abs() > 1e-6).nonzero(as_tuple=True)
    c = C[o, i, j]

    def f(x, gate):
        H = x.shape[2] // 2
        t = x[:, i, :H] * x[:, j, H:] * c.view(1, -1, 1)  # [N, nnz, H]
        out = x.new_zeros(x.shape[0], 9, H).index_add_(1, o, t)
        return out * gate

    return f, int(c.numel())


def per_output_factory(prod):
    C = prod.coupling(torch.float32, "cpu").reshape(9, 9, 9)
    blocks = []
    for o in range(9):
        nz = (C[o].abs() > 1e-6).nonzero()
        blocks.append((nz[:, 0], nz[:, 1], C[o][nz[:, 0], nz[:, 1]]))

    def f(x, gate):
        H = x.shape[2] // 2
        outs = []
        for I, J, c in blocks:
            t = x[:, I, :H] * x[:, J, H:]  # [N, k, H]
            outs.append(torch.einsum("k,nkh->nh", c, t))
        return torch.stack(outs, 1) * gate

    return f


def e3nn_factory(channels):
    from e3nn import o3

    irr = o3.Irreps(f"{channels}x0e+{channels}x1o+{channels}x2e")
    ins = []
    sl = {0: (0, 0), 1: (1, 1), 2: (2, 2)}
    for l1 in range(3):
        for l2 in range(3):
            for l3 in range(abs(l1 - l2), min(2, l1 + l2) + 1):
                if (l1 + l2 + l3) % 2 == 0:
                    ins.append((l1, l2, l3, "uuu", False))
    tp = o3.TensorProduct(irr, irr, irr, ins, shared_weights=True, internal_weights=False)

    def f(x, gate):
        # layout: [N, 9, 2H] -> irreps layout [N, H*(1+3+5)] with channel-major blocks per l
        H = x.shape[2] // 2
        a = torch.cat([x[:, 0:1, :H].transpose(1, 2).reshape(-1, H), x[:, 1:4, :H].transpose(1, 2).reshape(-1, 3 * H), x[:, 4:9, :H].transpose(1, 2).reshape(-1, 5 * H)], 1)
        b = torch.cat([x[:, 0:1, H:].transpose(1, 2).reshape(-1, H), x[:, 1:4, H:].transpose(1, 2).reshape(-1, 3 * H), x[:, 4:9, H:].transpose(1, 2).reshape(-1, 5 * H)], 1)
        w = torch.ones(tp.weight_numel)
        return tp(a, b, w)

    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=512)
    ap.add_argument("--channels", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=7)
    a = ap.parse_args()
    torch.manual_seed(0)
    grid = SO3Grid(2, 2)
    T, P = grid.to_grid_mat, grid.from_grid_mat
    N, H = a.nodes, a.channels
    x = torch.randn(N, 9, 2 * H)
    gate = torch.rand(N, 1, H)

    def grid_route(x, gate):
        g1 = torch.einsum("gi,nih->ngh", T, x[..., :H])
        g2 = torch.einsum("gi,nih->ngh", T, x[..., H:])
        return torch.einsum("og,ngh->noh", P, g1 * g2) * gate

    prod = aja.GatedGridProduct(grid)
    indexed, nnz = indexed_factory(prod)
    variants = {"grid": grid_route, "dense": prod, "indexed": indexed, "per-output": per_output_factory(prod)}
    try:
        variants["e3nn-cg"] = e3nn_factory(H)
    except Exception as exc:  # e3nn optional
        print("e3nn variant skipped:", repr(exc)[:80])
    ref = variants["grid"](x, gate)
    print(f"nodes={N} channels={H} threads={torch.get_num_threads()}  structural non-zeros: {nnz}/729")
    ts = {k: [] for k in variants}
    for r in range(a.repeats + 1):
        for k, f in variants.items():
            with torch.no_grad():
                t = time.perf_counter()
                out = f(x, gate)
                dt = time.perf_counter() - t
            if r:
                ts[k].append(dt)
            if r == 0 and k not in ("grid", "e3nn-cg"):
                err = float((out - ref).norm() / ref.norm())
                assert err < 1e-5, (k, err)
    base = statistics.median(ts["grid"])
    for k, v in ts.items():
        med = statistics.median(v)
        note = "  (different numbers: timing only)" if k == "e3nn-cg" else ""
        print(f"  {k:11s} {med * 1e3:8.2f} ms  {base / med:5.2f}x vs grid   spread {min(v) * 1e3:.1f}-{max(v) * 1e3:.1f} ms{note}")


if __name__ == "__main__":
    main()
