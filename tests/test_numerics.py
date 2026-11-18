"""Numerical validation of the coupling replacement beyond the small smoke configuration.

The contraction is algebraically exact, so the questions here are about finite precision and about state:
other channel counts and spherical-harmonic orders, tiny and huge activations, float64, gradients, long training
trajectories, stale caches, grid dropout, and the released (production-size) configuration.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import pytest
import torch

pytest.importorskip("atom_jepa")
import atom_jepa_accel as aja
from atom_jepa.conformers import smiles_to_samples
from atom_jepa.data.collate import GraphCollator
from atom_jepa.models.equiformer_v3.so3 import SO3Grid
from atom_jepa.models.jepa_equiformer import EquiformerV3Config, EquiformerV3Encoder
from mols import SMILES

GRIDS = [(1, 1), (2, 2), (3, 3), (3, 2)]


def reference(grid, x, gate):
    """Upstream's grid round trip in float64: from_grid( to_grid(a) * to_grid(b) ) * gate."""
    T, P = grid.to_grid_mat.double(), grid.from_grid_mat.double()
    x, gate = x.double(), gate.double()
    H = x.shape[2] // 2
    g1, g2 = torch.einsum("gi,nih->ngh", T, x[..., :H]), torch.einsum("gi,nih->ngh", T, x[..., H:])
    return torch.einsum("og,ngh->noh", P, g1 * g2) * gate


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300))


def _inputs(grid, N, H, scale, dtype, seed=0, grad=False):
    g = torch.Generator().manual_seed(seed)
    S = grid.to_grid_mat.shape[1]
    x = (torch.randn(N, S, 2 * H, generator=g, dtype=torch.float64) * scale).to(dtype)
    gate = torch.rand(N, 1, H, generator=g, dtype=torch.float64).to(dtype)
    return x.requires_grad_(grad), gate.requires_grad_(grad)


@pytest.mark.parametrize("lm", GRIDS, ids=lambda t: f"lmax{t[0]}-mmax{t[1]}")
@pytest.mark.parametrize("H", [1, 7, 64, 256])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_coupling_matches_grid_roundtrip(lm, H, scale):
    grid = SO3Grid(*lm)
    prod = aja.GatedGridProduct(grid)
    for dtype, tol in ((torch.float32, 2e-5), (torch.float64, 1e-12)):
        x, gate = _inputs(grid, 5, H, scale, dtype)
        assert rel(prod(x, gate), reference(grid, x, gate)) < tol, (dtype, tol)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_gradients_match_grid_roundtrip_float64(scale):
    grid = SO3Grid(2, 2)
    prod = aja.GatedGridProduct(grid)
    x, gate = _inputs(grid, 6, 16, scale, torch.float64, grad=True)
    w = torch.randn(6, grid.to_grid_mat.shape[1], 16, dtype=torch.float64)
    gx, gg = torch.autograd.grad((prod(x, gate) * w).sum(), (x, gate))
    rx, rg = torch.autograd.grad((reference(grid, x, gate) * w).sum(), (x, gate))
    assert rel(gx, rx) < 1e-11 and rel(gg, rg) < 1e-11


def test_float64_input_uses_the_float64_coupling():
    grid = SO3Grid(2, 2)
    prod = aja.GatedGridProduct(grid)
    x, gate = _inputs(grid, 4, 8, 1.0, torch.float64)
    assert prod.coupling(torch.float64, "cpu").dtype == torch.float64
    assert rel(prod(x, gate), reference(grid, x, gate)) < 1e-13  # float32 coupling would give ~1e-7


def test_nonzero_pattern_obeys_gaunt_selection_rules():
    """C is the Gaunt coupling: entries outside |l1-l2| <= l3 <= l1+l2, l1+l2+l3 even are float32 noise (~1e-7)."""
    grid = SO3Grid(2, 2)
    C = aja.GatedGridProduct(grid).coupling(torch.float64, "cpu").reshape(9, 9, 9)  # [o, i, j]
    ell = [0, 1, 1, 1, 2, 2, 2, 2, 2]
    allowed = torch.zeros(9, 9, 9, dtype=torch.bool)
    for o in range(9):
        for i in range(9):
            for j in range(9):
                l1, l2, l3 = ell[i], ell[j], ell[o]
                allowed[o, i, j] = (l1 + l2 + l3) % 2 == 0 and abs(l1 - l2) <= l3 <= l1 + l2
    assert float(C[~allowed].abs().max()) < 1e-6  # the grid matrices are stored in float32
    assert float(C[allowed].abs().max()) > 0.1
    assert (
        int((C[allowed].abs() > 1e-6).sum()) == 83
    )  # structural non-zeros of 729 (86 with the noise > 1e-7)


def test_coupling_is_rebuilt_when_grid_matrices_change():
    grid = SO3Grid(2, 2)
    prod = aja.GatedGridProduct(grid)
    x, gate = _inputs(grid, 4, 8, 1.0, torch.float32)
    before = prod(x, gate)
    with torch.no_grad():
        grid.to_grid_mat.mul_(1.5)  # in-place edit of a grid buffer
    assert rel(prod(x, gate), reference(grid, x, gate)) < 2e-5
    assert rel(prod(x, gate), before) > 0.1
    grid.from_grid_mat = grid.from_grid_mat * 2.0  # buffer replaced
    assert rel(prod(x, gate), reference(grid, x, gate)) < 2e-5
    grid.double()  # module dtype change
    assert rel(prod(x, gate), reference(grid, x, gate)) < 2e-5


def test_input_shape_is_validated():
    prod = aja.GatedGridProduct(SO3Grid(2, 2))
    with pytest.raises(ValueError, match="expected"):
        prod(torch.randn(3, 4, 8), torch.ones(3, 1, 4))


# ---- whole-encoder behaviour ------------------------------------------------------------------------------------


def _cfg(dropout=0.0, **kw):
    base = dict(
        num_channels=64,
        num_layers=2,
        lmax=2,
        mmax=2,
        max_radius=6.0,
        num_radial_basis=32,
        num_heads=4,
        attn_hidden_channels=32,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=128,
        edge_channels=32,
        attn_grid_resolution_list=[14, 8],
        ffn_grid_resolution_list=[14, 14],
        grad_checkpointing=False,
        dropout=dropout,
    )
    base.update(kw)
    return EquiformerV3Config(**base)


def _enc(cfg, seed=0):
    torch.manual_seed(seed)
    return EquiformerV3Encoder(cfg)


def _hooks(enc):
    return [
        getattr(o, "gated_product", None)
        for b in enc.body.blocks
        for o in (b.ga.act, b.ffn.grid_mlp)
    ]


def _ffn_hooks(enc):
    return [getattr(b.ffn.grid_mlp, "gated_product", None) for b in enc.body.blocks]


def _act_hooks(enc):
    return [getattr(b.ga.act, "gated_product", None) for b in enc.body.blocks]


def assert_unaccelerated(enc):
    assert all(h is None for h in _hooks(enc)), "reference encoder must carry no acceleration hooks"
    assert not any(hasattr(b, "_aja_hook") for b in enc.body.blocks)
    assert not any(b.ffn.optimize_grid_mlp for b in enc.body.blocks)  # upstream's CPU default


def assert_accelerated(enc):
    assert all(isinstance(h, aja.GatedGridProduct) for h in _hooks(enc))
    assert all(b.ffn.optimize_grid_mlp for b in enc.body.blocks)


@pytest.fixture(autouse=True)
def _restore_activation():
    yield
    aja.disable()
    aja.enable()


def build_pair(cfg, mode="eval"):
    """(reference, accelerated) with identical weights; the reference is proven unaccelerated.

    Order matters because the package patches model construction: disable first, build and copy weights, check that
    neither encoder is accelerated, then accelerate only the second and re-check both.
    """
    aja.disable()
    ref, new = _enc(cfg), _enc(cfg)
    new.load_state_dict(ref.state_dict())
    for e in (ref, new):
        getattr(e, mode)()
        assert_unaccelerated(e)
    assert aja.accelerate(new)
    assert_unaccelerated(ref)
    return ref, new


@pytest.fixture(scope="module")
def batch():
    return GraphCollator(cutoff=6.0)([smiles_to_samples(s)[0] for s in SMILES[:3]])


def test_training_with_grid_dropout_keeps_working(batch):
    """Upstream's hook contract forbids grid dropout in training; accelerate() must not turn a model that trains into a crash."""
    ref, new = build_pair(_cfg(0.1), "train")
    torch.manual_seed(5)
    out_ref = ref(batch)
    torch.manual_seed(5)
    out_new = new(batch)  # would raise RuntimeError if the hook stayed installed on dropout layers
    out_new.pow(2).sum().backward()
    assert all(
        h is None for h in _ffn_hooks(new)
    )  # FFN grid dropout in training: upstream's own path
    assert all(
        isinstance(h, aja.GatedGridProduct) for h in _act_hooks(new)
    )  # attention activation has none
    # same seed -> same dropout masks (upstream's optimised path keeps the legacy mask layout)
    torch.testing.assert_close(out_new, out_ref, atol=1e-4, rtol=1e-3)


def test_grid_dropout_policy_zero_train_eval(batch):
    # (1) no dropout: hook live in both modes
    ref, new = build_pair(_cfg(0.0), "train")
    new(batch)
    assert_accelerated(new)
    new.eval()
    with torch.no_grad():
        out = new(batch)
    assert_accelerated(new)
    torch.testing.assert_close(out, ref.eval()(batch), atol=1e-4, rtol=1e-3)
    # (2) dropout>0 while training: hook off; (3) the same model in eval: hook on, outputs match the original
    ref, new = build_pair(_cfg(0.1), "train")
    new(batch)
    assert all(h is None for h in _ffn_hooks(new))
    assert all(isinstance(h, aja.GatedGridProduct) for h in _act_hooks(new))
    new.eval()
    ref.eval()
    with torch.no_grad():
        out_new, out_ref = new(batch), ref(batch)
    assert_accelerated(new)  # eval-mode inference of a model configured with dropout is accelerated
    torch.testing.assert_close(out_new, out_ref, atol=1e-4, rtol=1e-3)
    new.train()
    new(batch)  # and it toggles back
    assert all(h is None for h in _ffn_hooks(new))


def test_dropout_p0_module_in_activation_is_handled(batch):
    """Upstream raises for any non-Identity grid_drop in training, even Dropout(p=0); we must not reinstall the crash."""
    ref, new = build_pair(_cfg(0.0), "train")
    for e in (ref, new):
        for b in e.body.blocks:
            b.ga.act.grid_drop = torch.nn.Dropout(0.0)
    new(batch)  # would raise if the activation hook were live while training
    assert all(getattr(b.ga.act, "gated_product", None) is None for b in new.body.blocks)
    assert all(
        isinstance(b.ffn.grid_mlp.gated_product, aja.GatedGridProduct) for b in new.body.blocks
    )
    new.eval()
    with torch.no_grad():
        new(batch)
    assert all(isinstance(b.ga.act.gated_product, aja.GatedGridProduct) for b in new.body.blocks)


def test_unaccelerate_removes_hooks_and_toggle():
    _ref, new = build_pair(_cfg(0.1), "train")
    aja.unaccelerate(new)
    assert_unaccelerated(new)


def test_training_trajectory_tracks_original(batch):
    """25 Adam steps: the accelerated and original models stay together (float32 rounding only)."""
    ref, new = build_pair(_cfg(0.0), "train")
    opts = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in (ref, new)]
    drift = []
    for _step in range(25):
        losses = []
        for m, o in zip((ref, new), opts):
            o.zero_grad()
            loss = m(batch).pow(2).mean()
            loss.backward()
            o.step()
            losses.append(loss.item())
        drift.append(abs(losses[1] - losses[0]) / abs(losses[0]))
    w = [torch.cat([p.detach().flatten() for p in m.parameters()]) for m in (ref, new)]
    assert max(drift) < 1e-4, drift
    assert rel(w[1], w[0]) < 1e-4


@pytest.mark.slow
def test_production_size_matches_original(batch):
    """The released configuration (256 channels, 8 layers): forward, then gradients."""
    cfg = _cfg(
        num_channels=256,
        num_layers=8,
        num_radial_basis=128,
        num_heads=8,
        attn_hidden_channels=64,
        attn_alpha_channels=64,
        ffn_hidden_channels=512,
        edge_channels=128,
    )
    ref, new = build_pair(cfg, "eval")
    one = GraphCollator(cutoff=6.0)([smiles_to_samples(SMILES[0])[0]])
    with torch.no_grad():
        assert rel(new(one), ref(one)) < 1e-5
    grads = []
    for m in (ref, new):
        m.zero_grad()
        m(one).pow(2).sum().backward()
        grads.append(torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]))
    assert rel(grads[1], grads[0]) < 1e-4
