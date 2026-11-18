import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import pytest
import torch

pytest.importorskip("atom_jepa")
import atom_jepa_accel as aja
from atom_jepa.models.jepa_equiformer import EquiformerV3Config, EquiformerV3Encoder
from atom_jepa.conformers import smiles_to_samples
from atom_jepa.data.collate import GraphCollator
from mols import SMILES


def make_encoder(layers=2, seed=0):
    cfg = EquiformerV3Config(
        num_channels=64,
        num_layers=layers,
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
    )
    torch.manual_seed(seed)
    return EquiformerV3Encoder(cfg)


@pytest.fixture(scope="module")
def batch():
    return GraphCollator(cutoff=6.0)([smiles_to_samples(s)[0] for s in SMILES[:5]])


def test_forward_equal(batch):
    ref_enc, new_enc = make_encoder().eval(), make_encoder().eval()
    new_enc.load_state_dict(ref_enc.state_dict())
    with torch.no_grad():
        ref = ref_enc(batch)
    assert aja.accelerate(new_enc) is True
    with torch.no_grad():
        out = new_enc(batch)
    assert (out - ref).norm() / ref.norm() < 1e-5


def test_gradients_equal(batch):
    ref_enc, new_enc = make_encoder().train(), make_encoder().train()
    new_enc.load_state_dict(ref_enc.state_dict())
    for e in (ref_enc, new_enc):
        for m in e.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
    aja.accelerate(new_enc)
    for e in (ref_enc, new_enc):
        e.zero_grad()
        e(batch).pow(2).sum().backward()
    for (n, a), (_, b) in zip(ref_enc.named_parameters(), new_enc.named_parameters()):
        if a.grad is None:
            assert b.grad is None
            continue
        torch.testing.assert_close(b.grad, a.grad, atol=1e-4, rtol=1e-3, msg=n)


def test_state_dict_and_params_unchanged():
    a, b = make_encoder(), make_encoder()
    aja.accelerate(b)
    assert list(a.state_dict()) == list(b.state_dict())
    assert [p.shape for p in a.parameters()] == [p.shape for p in b.parameters()]


def test_unaccelerate_restores(batch):
    e = make_encoder().eval()
    with torch.no_grad():
        ref = e(batch)
    aja.accelerate(e)
    aja.unaccelerate(e)
    assert all(getattr(b.ga.act, "gated_product", None) is None for b in e.body.blocks)
    e.body.set_grid_mlp_optimization(False)
    with torch.no_grad():
        torch.testing.assert_close(e(batch), ref)


def test_setter_patch_installs_on_cpu(batch):
    aja.enable()
    e = make_encoder().eval()
    e.body.set_grid_mlp_optimization(True)
    assert all(isinstance(b.ga.act.gated_product, aja.GatedGridProduct) for b in e.body.blocks)
    aja.disable()
    e2 = make_encoder()
    e2.body.set_grid_mlp_optimization(True)
    assert all(getattr(b.ga.act, "gated_product", None) is None for b in e2.body.blocks)
    aja.enable()


def test_status_and_idempotent():
    s1 = aja.enable()
    s2 = aja.enable()
    assert s1 == s2 and s1["enabled"]


def test_does_not_override_existing_hook():
    e = make_encoder()
    sentinel = object()
    e.body.blocks[0].ga.act.gated_product = sentinel
    aja.accelerate(e)
    assert e.body.blocks[0].ga.act.gated_product is sentinel
