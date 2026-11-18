"""atom_jepa_accel under scbirlab/aspect's `atom-jepa` featuriser (real Aspect + real AtomJEPA, random weights)."""

import numpy as np
import pytest

pytest.importorskip("aspect")
pytest.importorskip("schemist")
import atom_jepa_accel as aja
from mols import SMILES

MOLS = SMILES[:10] + [None]


def conformer_rows(n_conformers=1, mols=MOLS):
    from aspect.transform.conformers import Conformers

    return Conformers(n_conformers=n_conformers)({"smiles": mols}, "smiles")


def embed(aa, rows, accel, **kw):
    """Fresh encoder each call (Aspect memoizes it), with accel on or off at load time."""
    (aja.enable if accel else aja.disable)()
    aa._load_atom_jepa_cached.cache_clear()
    try:
        return aa.AtomJEPAEmbedding(device="cpu", **kw)({"c": rows}, "c")
    finally:
        aja.enable()


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize(
    "kw", [dict(), dict(model="molecules"), dict(layers="all"), dict(layers=3), dict(batch_size=3)]
)
def test_embeddings_match_original(aspect_atom_jepa, kw):
    rows = conformer_rows()
    ref = embed(aspect_atom_jepa, rows, False, **kw)
    new = embed(aspect_atom_jepa, rows, True, **kw)
    assert new.shape == ref.shape and rel(new, ref) < 1e-5
    assert not new[-1].any()  # missing molecule -> zero vector


def test_multiple_conformers_span_batches(aspect_atom_jepa):
    rows = conformer_rows(3, SMILES[:5])
    ref = embed(aspect_atom_jepa, rows, False, batch_size=2)
    new = embed(aspect_atom_jepa, rows, True, batch_size=2)
    assert rel(new, ref) < 1e-5


def test_accel_really_active_in_aspect_loaded_encoder(aspect_atom_jepa):
    aja.enable()
    aspect_atom_jepa._load_atom_jepa_cached.cache_clear()
    enc = aspect_atom_jepa._load_atom_jepa(
        "bio", aspect_atom_jepa.ATOM_JEPA_REVISION, "cpu", "fp32"
    )
    assert all(
        isinstance(b.ga.act.gated_product, aja.GatedGridProduct) for b in enc.encoder.body.blocks
    )
    assert not any(p.requires_grad for p in enc.parameters())  # still frozen
    aja.disable()
    aspect_atom_jepa._load_atom_jepa_cached.cache_clear()
    enc = aspect_atom_jepa._load_atom_jepa(
        "bio", aspect_atom_jepa.ATOM_JEPA_REVISION, "cpu", "fp32"
    )
    assert all(getattr(b.ga.act, "gated_product", None) is None for b in enc.encoder.body.blocks)
    aja.enable()


def test_encoder_loaded_before_import_is_retrofitted_on_request(aspect_atom_jepa):
    aja.disable()
    aspect_atom_jepa._load_atom_jepa_cached.cache_clear()
    enc = aspect_atom_jepa._load_atom_jepa(
        "bio", aspect_atom_jepa.ATOM_JEPA_REVISION, "cpu", "fp32"
    )
    rows = conformer_rows(mols=SMILES[:4])
    before = aspect_atom_jepa.AtomJEPAEmbedding(device="cpu")(
        {"c": rows}, "c"
    )  # memoized, unaccelerated
    assert getattr(enc.encoder.body.blocks[0].ga.act, "gated_product", None) is None
    st = aja.enable()  # import happens "late": existing encoders are NOT touched implicitly ...
    assert st["accelerated_encoders"] == 0
    assert getattr(enc.encoder.body.blocks[0].ga.act, "gated_product", None) is None
    assert aja.accelerate_existing() >= 1  # ... only on request
    assert aja.status()["accelerated_encoders"] >= 1
    after = aspect_atom_jepa.AtomJEPAEmbedding(device="cpu")(
        {"c": rows}, "c"
    )  # same memoized encoder
    assert rel(after, before) < 1e-5
    aja.disable()  # fully restored
    assert getattr(enc.encoder.body.blocks[0].ga.act, "gated_product", None) is None
    assert not enc.encoder.body.blocks[0].ffn.optimize_grid_mlp  # upstream CPU default
    aja.enable()


def test_native_smiles_and_sample_inputs(aspect_atom_jepa):
    """AtomJEPA.embed accepts SMILES (upstream generates conformers) as well as pre-generated samples."""
    from atom_jepa.conformers import smiles_to_samples

    out = {}
    for accel in (False, True):
        (aja.enable if accel else aja.disable)()
        aspect_atom_jepa._load_atom_jepa_cached.cache_clear()
        enc = aspect_atom_jepa._load_atom_jepa(
            "bio", aspect_atom_jepa.ATOM_JEPA_REVISION, "cpu", "fp32"
        )
        samples = [smiles_to_samples(s)[0] for s in SMILES[:4]]
        out[accel] = (enc.embed(SMILES[:4]).numpy(), enc.embed(samples).numpy())
    aja.enable()
    for a, b in zip(out[True], out[False]):
        assert rel(a, b) < 1e-5
    np.testing.assert_allclose(
        out[True][0], out[True][1], rtol=1e-5, atol=1e-6
    )  # SMILES == pre-generated
