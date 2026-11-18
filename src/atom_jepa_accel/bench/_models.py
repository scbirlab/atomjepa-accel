"""Random-weight Atom-JEPA encoder (released configuration, or a small one for gradient checks)."""

from __future__ import annotations

import torch


def make_encoder(small: bool = False, seed: int = 0):
    from atom_jepa.models.jepa_equiformer import EquiformerV3Config, EquiformerV3Encoder

    if small:
        cfg = EquiformerV3Config(
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
        )
    else:
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
    torch.manual_seed(seed)
    return EquiformerV3Encoder(cfg).eval()
