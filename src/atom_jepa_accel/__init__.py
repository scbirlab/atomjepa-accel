"""atom_jepa_accel -- drop-in CPU acceleration for Atom-JEPA (EquiformerV3 encoder).

    import atom_jepa_accel      # activates verified optimisations
    import atom_jepa            # ordinary code, unchanged

* enables the grid-MLP path on CPU (upstream enables it only on CUDA);
* replaces the S2-grid round trip in every gated activation by an exactly equivalent sparse
  trilinear coupling (precomputed in float64 from the model's own grid matrices);
* no parameter, buffer or state_dict key changes.

Set ATOM_JEPA_ACCEL_DISABLE=1 to prevent automatic activation.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import os
import warnings
import weakref
from typing import Dict

import torch

__version__ = "0.1.0"
KNOWN_HASHES = {
    "HeadlessEquiformerV3.set_grid_mlp_optimization": ["fc3058de9e77663e"],
    "AtomJEPA._configure_execution": ["4a23f327128b82e6"],
}
_state: Dict[str, dict] = {}
_accelerated: weakref.WeakSet = weakref.WeakSet()  # bodies we modified, so disable() can undo it


class GatedGridProduct:
    """Exact replacement of the grid round trip  from_grid( to_grid(a) * to_grid(b) ) * gate.

    C[o,i,j] = sum_g P[o,g] T[g,i] T[g,j] is built once in float64; it has ~85 non-zeros of 729,
    which a dense GEMM over the flattened outer product exploits well on CPU. A plain callable
    (not an nn.Module) so state_dict and parameter lists are untouched.
    """

    def __init__(self, grid):
        T = grid.to_grid_mat.detach().double()
        P = grid.from_grid_mat.detach().double()
        C = torch.einsum("og,gi,gj->oij", P, T, T)
        self.S = T.shape[1]
        self.C = C.reshape(self.S, self.S * self.S).float()

    def __call__(self, x, gate):
        H = x.shape[2] // 2
        N = x.shape[0]
        outer = (x[..., :H].unsqueeze(2) * x[..., H:].unsqueeze(1)).reshape(N, self.S * self.S, H)
        return torch.matmul(self.C.to(x.dtype), outer) * gate


def _on_cpu(body) -> bool:
    return next(body.parameters()).device.type == "cpu"


def _body_of(model_or_body):
    body = getattr(model_or_body, "encoder", model_or_body)
    return getattr(body, "body", body)


def accelerate(model_or_body, force: bool = False, _prev=None) -> bool:
    """Install the optimisations on an already-built model/encoder/body. Returns True if applied.

    Accepts AtomJEPA, an encoder with ``.body``, or a HeadlessEquiformerV3. CPU only (on CUDA the
    upstream fused kernels are kept). Idempotent; reversible with :func:`unaccelerate`.
    """
    body = _body_of(model_or_body)
    if not hasattr(body, "blocks") or not (force or _on_cpu(body)):
        return False
    for i, blk in enumerate(body.blocks):
        for owner, grid in (
            (blk.ga.act, blk.ga.act.so3_grid),
            (blk.ffn.grid_mlp, blk.ffn.so3_grid),
        ):
            if getattr(owner, "gated_product", None) is None:  # never override cuEq / ROCm hooks
                owner.gated_product = GatedGridProduct(grid)
                owner._aja_installed = True
        if not hasattr(blk.ffn, "_aja_prev_optimize"):
            blk.ffn._aja_prev_optimize = bool(
                blk.ffn.optimize_grid_mlp if _prev is None else _prev[i]
            )
        blk.ffn.optimize_grid_mlp = True
    _accelerated.add(body)
    return True


def unaccelerate(model_or_body) -> None:
    """Undo :func:`accelerate`, restoring each block's original grid-MLP flag."""
    body = _body_of(model_or_body)
    for blk in body.blocks:
        for owner in (blk.ga.act, blk.ffn.grid_mlp):
            if getattr(owner, "_aja_installed", False):
                owner.gated_product = None
                owner._aja_installed = False
        if hasattr(blk.ffn, "_aja_prev_optimize"):
            blk.ffn.optimize_grid_mlp = blk.ffn._aja_prev_optimize
            del blk.ffn._aja_prev_optimize
    _accelerated.discard(body)


def _auto_accelerate(body, prev=None) -> bool:
    """accelerate() for automatic paths: never raises into user code (e.g. upstream layout drift)."""
    try:
        return accelerate(body, _prev=prev)
    except Exception as exc:
        _state["auto_error"] = repr(exc)
        return False


def _retrofit_existing() -> int:
    """Accelerate encoders that were built before this package was imported (CPU only)."""
    try:
        from atom_jepa.models.eqv3_backbone import HeadlessEquiformerV3 as Body
    except Exception:
        return 0
    n = 0
    with warnings.catch_warnings():  # some objects emit deprecation warnings when merely touched
        warnings.simplefilter("ignore")
        for obj in gc.get_objects():
            try:
                # type() rather than isinstance(): never calls __getattr__/__class__ on arbitrary objects
                if (
                    issubclass(type(obj), Body)
                    and obj not in _accelerated
                    and _auto_accelerate(obj)
                ):
                    n += 1
            except Exception:  # an object that is mid-construction or odd: leave it alone
                continue
    return n


def _digest(fn) -> str:
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:16]


def enable(retrofit: bool = True) -> Dict:
    """Patch atom_jepa so models are accelerated on CPU automatically (idempotent).

    With ``retrofit=True`` (default) encoders that already exist, for example an encoder an application
    loaded and memoized before importing this package, are accelerated too.
    """
    try:
        from atom_jepa.api import AtomJEPA
        from atom_jepa.models.eqv3_backbone import HeadlessEquiformerV3 as Body
    except Exception as exc:
        return {"enabled": False, "reason": f"cannot import atom_jepa: {exc!r}"}

    if "setter" not in _state:
        orig = Body.set_grid_mlp_optimization
        d = _digest(orig)
        if d in KNOWN_HASHES["HeadlessEquiformerV3.set_grid_mlp_optimization"]:

            def set_grid_mlp_optimization(self, enabled=True, _orig=orig):
                prev = [bool(b.ffn.optimize_grid_mlp) for b in self.blocks]
                _orig(self, enabled)
                if enabled and _on_cpu(self):
                    _auto_accelerate(self, prev)
                elif not enabled and self in _accelerated:
                    unaccelerate(self)
                    _orig(self, False)

            set_grid_mlp_optimization._aja = True
            Body.set_grid_mlp_optimization = set_grid_mlp_optimization
            _state["setter"] = {"cls": Body, "orig": orig}
        else:
            _state["setter_skipped"] = d
    if "cfg" not in _state:
        orig = AtomJEPA._configure_execution
        d = _digest(orig)
        if d in KNOWN_HASHES["AtomJEPA._configure_execution"]:

            def _configure_execution(self, device, precision, cuequivariance, compile, _orig=orig):
                out = _orig(self, device, precision, cuequivariance, compile)
                if torch.device(device).type == "cpu":
                    self.encoder.body.set_grid_mlp_optimization(True)
                return out

            AtomJEPA._configure_execution = _configure_execution
            _state["cfg"] = {"cls": AtomJEPA, "orig": orig}
        else:
            _state["cfg_skipped"] = d
    if retrofit and "setter" in _state:  # only on a verified upstream layout
        _state["retrofitted"] = _retrofit_existing()
    return status()


def disable() -> Dict:
    """Restore the original methods and undo the changes made to any encoders (weights untouched)."""
    for body in list(_accelerated):
        unaccelerate(body)
    _state.pop("retrofitted", None)
    for key, attr in (("setter", "set_grid_mlp_optimization"), ("cfg", "_configure_execution")):
        if key in _state:
            setattr(_state[key]["cls"], attr, _state[key]["orig"])
            del _state[key]
    return status()


def status() -> Dict:
    return {
        "enabled": "setter" in _state or "cfg" in _state,
        "atom_jepa_accel": __version__,
        "torch": torch.__version__,
        "patches": {
            "set_grid_mlp_optimization": "setter" in _state,
            "_configure_execution": "cfg" in _state,
        },
        "accelerated_encoders": len(_accelerated),
        "skipped_unverified": {k: v for k, v in _state.items() if k.endswith("_skipped")},
        "auto_error": _state.get("auto_error"),
    }


if os.environ.get("ATOM_JEPA_ACCEL_DISABLE", "0") != "1":
    enable()
