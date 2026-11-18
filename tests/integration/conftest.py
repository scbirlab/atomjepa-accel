"""Fake Hugging Face hub (released checkpoint layout, random weights) for offline tests of scbirlab/aspect."""

import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import torch


def released_config():
    from atom_jepa.models.jepa_equiformer import EquiformerV3Config

    return EquiformerV3Config(
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


@pytest.fixture(scope="session")
def fake_atom_jepa_hub(tmp_path_factory):
    pytest.importorskip("aspect")
    pytest.importorskip("atom_jepa")
    from safetensors.torch import save_file
    from atom_jepa.models.jepa_equiformer import EquiformerV3Encoder

    root = tmp_path_factory.mktemp("atom_jepa_hub")
    cfg = released_config()
    for name, seed in (("bio", 1), ("molecules", 2)):
        torch.manual_seed(seed)
        enc = EquiformerV3Encoder(cfg)
        (root / name).mkdir()
        save_file(
            {k: v.detach().clone().contiguous() for k, v in enc.state_dict().items()},
            str(root / name / "model.safetensors"),
        )
        (root / name / "config.json").write_text(json.dumps({"eqv3_cfg": dataclasses.asdict(cfg)}))
    return root


@pytest.fixture()
def aspect_atom_jepa(fake_atom_jepa_hub, monkeypatch):
    import huggingface_hub
    import aspect.transform.atomjepa as aa

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda repo, name, **kw: str(fake_atom_jepa_hub / name)
    )
    aa._load_atom_jepa_cached.cache_clear()
    yield aa
    aa._load_atom_jepa_cached.cache_clear()
