# Quickstart

```python
import atom_jepa_accel  # activates automatically (ATOM_JEPA_ACCEL_DISABLE=1 to prevent)
import atom_jepa

print(atom_jepa_accel.status())  # enabled?, versions, which patches are active and why not
atom_jepa_accel.disable()  # restores the original functions (parameters untouched)
atom_jepa_accel.enable()  # idempotent
```

## Models you built yourself

```python
import atom_jepa_accel as aja

aja.accelerate(model)  # AtomJEPA, an encoder, or the EquiformerV3 body; CPU only; idempotent
aja.unaccelerate(model)  # restores everything, including the previous grid-MLP flag
```
Models loaded with `AtomJEPA.from_pretrained(..., device="cpu")` after the import are accelerated automatically; models
that already exist when you import are retrofitted by `enable()`.
