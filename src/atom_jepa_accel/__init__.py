"""atom_jepa_accel -- drop-in CPU acceleration for Atom-JEPA (EquiformerV3 encoder).

    import atom_jepa_accel      # activates verified optimisations
    import atom_jepa            # ordinary code, unchanged

* enables the grid-MLP path on CPU (upstream enables it only on CUDA);
* replaces the S2-grid round trip in every gated activation by an exactly equivalent trilinear coupling
  (the Gaunt coupling upstream already uses for cuEquivariance/ROCm), precomputed in float64 from the model's
  own grid matrices and applied as one dense GEMM;
* layers with grid dropout use the coupling in eval mode and upstream's own path while training (its hook
  contract forbids dropout there); a forward pre-hook on each block switches between the two;
* no parameter, buffer or state_dict key changes.

Set ATOM_JEPA_ACCEL_DISABLE=1 to prevent automatic activation. Models built before the import are only
accelerated on request: ``accelerate(model)``, ``accelerate_existing()`` or ATOM_JEPA_ACCEL_RETROFIT=1.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import os
import warnings
import weakref
from typing import Dict, Optional

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

    C[o,i,j] = sum_g P[o,g] T[g,i] T[g,j]. This is the same coupling that upstream's own cuEquivariance and ROCm
    gated products use (``atom_jepa.models.equiformer_v3.admet_cue_ops``, ``atom_jepa.rocm_grid``); here it is
    applied as one dense GEMM over the flattened outer product, which is what suits CPUs (it has ~85 non-zeros of
    729, but ``experiments/coupling_alternatives`` shows indexed contraction is slower on CPU). It is the Gaunt
    tensor product of the two inputs, not new mathematics.

    The float64 master coupling is derived from the grid's own matrices and cached per ``(dtype, device)``.
    It is rebuilt automatically if those matrices are replaced or modified in place (version counter), so
    ``model.double()``, ``model.to(device)`` or edited grid buffers cannot leave it stale. float64 inputs use the
    float64 master (no float32 round trip). A plain callable (not an nn.Module): state_dict and parameter lists
    are untouched.
    """

    def __init__(self, grid):
        self._grid = weakref.ref(grid)
        self.S = grid.to_grid_mat.shape[1]
        self._source = None
        self._master = None
        self._cache: Dict[tuple, torch.Tensor] = {}
        self._rebuild(grid)

    @staticmethod
    def _source_key(grid):
        T, P = grid.to_grid_mat, grid.from_grid_mat
        return (id(T), T._version, tuple(T.shape), id(P), P._version, tuple(P.shape))

    def _rebuild(self, grid):
        T = grid.to_grid_mat.detach().cpu().double()
        P = grid.from_grid_mat.detach().cpu().double()
        if T.shape[1] != self.S:
            raise ValueError("grid changed its number of coefficients after acceleration")
        C = torch.einsum("og,gi,gj->oij", P, T, T)
        self._master = C.reshape(self.S, self.S * self.S).contiguous()
        self._source = self._source_key(grid)
        self._cache.clear()

    def coupling(self, dtype, device):
        """The (S, S*S) coupling in ``dtype`` on ``device``, rebuilt first if the grid matrices changed."""
        grid = self._grid()
        if grid is not None and self._source_key(grid) != self._source:
            self._rebuild(grid)
        key = (dtype, str(device))
        C = self._cache.get(key)
        if C is None:
            C = self._master.to(device=device, dtype=dtype)
            self._cache[key] = C
        return C

    def __call__(self, x, gate):
        if x.dim() != 3 or x.shape[1] != self.S or x.shape[2] % 2:
            raise ValueError(f"expected [nodes, {self.S}, 2*channels], got {tuple(x.shape)}")
        H = x.shape[2] // 2
        N = x.shape[0]
        outer = (x[..., :H].unsqueeze(2) * x[..., H:].unsqueeze(1)).reshape(N, self.S * self.S, H)
        return torch.matmul(self.coupling(x.dtype, x.device), outer) * gate


def _grid_dropout_active(owner) -> bool:
    """True if upstream's ``gated_product`` contract forbids the hook on this layer *while training*.

    Upstream raises ("cuEquivariance grid fusion requires inactive grid dropout") when a hook is installed and
    the layer is in training mode with grid dropout: the FFN grid MLP when ``dropout > 0``, the attention
    activation when ``grid_drop`` is anything but ``Identity`` (even ``Dropout(p=0)``). The restriction is
    behavioural, not structural: in eval mode the hook is fine.
    """
    drop = getattr(owner, "grid_drop", None)
    if drop is not None and not isinstance(drop, torch.nn.Identity):
        return True
    return float(getattr(owner, "dropout", 0.0) or 0.0) > 0.0


def _owners(blk):
    return (blk.ga.act, blk.ffn.grid_mlp)


def _toggle(blk, _args=None):
    """Forward pre-hook on a block: our hook is live except while training a layer with grid dropout.

    Such layers fall back to upstream's own (CPU-enabled) grid-MLP path in training and use the coupling in eval,
    so a model with dropout configured is accelerated for inference and still trains correctly.

    Mutable state: this assigns ``owner.gated_product`` on each forward. Sequential forwards are supported; concurrent
    forwards in the same mode have not been independently validated; concurrent forwards in different modes, or
    ``train()``/``eval()`` during a forward, are unsupported (see docs/compatibility.md).
    """
    for owner in _owners(blk):
        if getattr(owner, "_aja_installed", False):
            owner.gated_product = (
                None if (owner.training and _grid_dropout_active(owner)) else owner._aja_product
            )


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
            if getattr(owner, "_aja_installed", False):
                continue  # already ours (idempotent)
            if (
                getattr(owner, "gated_product", None) is not None
            ):  # never override cuEq / ROCm hooks
                continue
            owner._aja_product = GatedGridProduct(grid)
            owner._aja_installed = True
        if any(getattr(o, "_aja_installed", False) for o in _owners(blk)):
            if not hasattr(blk, "_aja_hook"):
                blk._aja_hook = blk.register_forward_pre_hook(_toggle)
            _toggle(blk)
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
        if hasattr(blk, "_aja_hook"):
            blk._aja_hook.remove()
            del blk._aja_hook
        for owner in _owners(blk):
            if getattr(owner, "_aja_installed", False):
                owner.gated_product = None
                owner._aja_installed = False
                owner.__dict__.pop("_aja_product", None)
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


def accelerate_existing() -> int:
    """Accelerate every Atom-JEPA encoder that already exists in this process (CPU only); returns how many.

    Explicit and opt-in because it scans ``gc.get_objects()``, which is too intrusive to do implicitly in a
    long-lived process. It is only needed when this package is imported *after* models were built and cached
    (for example by a program that memoizes its encoder). Programs started through ``python -m atom_jepa_accel``
    or that import this package first build every model accelerated and do not need it. To accelerate one
    specific model, call :func:`accelerate` on it instead.
    """
    return _retrofit_existing()


def _retrofit_existing() -> int:
    """gc-based scan behind :func:`accelerate_existing`."""
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


def enable(retrofit: Optional[bool] = None) -> Dict:
    """Patch atom_jepa so models built from now on are accelerated on CPU automatically (idempotent).

    Existing encoders are left alone unless ``retrofit=True`` (or ``ATOM_JEPA_ACCEL_RETROFIT=1`` for the
    automatic activation on import); see :func:`accelerate_existing`. ``retrofit=None`` means "use the
    environment variable".
    """
    if retrofit is None:
        retrofit = os.environ.get("ATOM_JEPA_ACCEL_RETROFIT", "0") == "1"
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
