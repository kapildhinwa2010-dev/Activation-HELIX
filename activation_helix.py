# Activation-HELIX - Activation matrices stored and recalled through HELIX
# Part of the Kapildhinwa2010-dev Toolkit.

"""Activation-HELIX: store layer activations in a HELIX-style wave store.

Lineage: DNA + HELIX (reserv.dean-part-2). The storage engine (HybridStore)
is shared with DNA-HELIX, which stores weight matrices; this module applies
the same interference-recall storage to ACTIVATION matrices and adds
SVD-truncated activation storage plus an optional PyTorch hook capture, so
recorded activations can be recalled by name instead of a filename.

Limitations (honest scope):
- Recall is by nearest stored key above a similarity threshold; two very
  similar layer names can collide.
- SVD truncation is lossy; retained variance is reported per entry.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["HybridStore", "RecallMiss"]


class RecallMiss(Exception):
    """No stored model resonates strongly enough with the probe."""


def _fourier_basis(n_cols: int, num_coeffs: int) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n_cols, endpoint=False)
    cols = [np.ones_like(t)]
    for k in range(1, num_coeffs + 1):
        cols.append(np.sin(k * t))
        cols.append(np.cos(k * t))
    return np.stack(cols, axis=1)


def _quantize(values: np.ndarray, bits: int):
    lo = float(values.min())
    hi = float(values.max())
    levels = 2 ** bits - 1
    scale = (hi - lo) / levels if hi > lo else 1.0
    codes = np.clip(np.round((values - lo) / scale), 0, levels)
    return codes.astype(np.uint16), lo, scale


def _dequantize(codes: np.ndarray, lo: float, scale: float) -> np.ndarray:
    return codes.astype(np.float64) * scale + lo


class HybridStore:
    """Store DNA-encoded matrices; recall them by key interference."""

    def __init__(self, num_coeffs: int = 8, bits: int = 8, dim: int = 32, threshold: float = 0.5) -> None:
        if num_coeffs < 1:
            raise ValueError("num_coeffs must be >= 1")
        if not 1 <= bits <= 16:
            raise ValueError("bits must be in [1, 16]")
        if dim < 4:
            raise ValueError("dim must be >= 4")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.num_coeffs = num_coeffs
        self.bits = bits
        self.dim = dim
        self.threshold = threshold
        self._names = []
        self._keys = []
        self._genes = []

    def _embed(self, name: str) -> np.ndarray:
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        vec = rng.standard_normal(self.dim)
        return vec / np.linalg.norm(vec)

    def _encode(self, weights: np.ndarray):
        W = np.asarray(weights, dtype=np.float64)
        if W.ndim != 2 or W.size == 0:
            raise ValueError("weights must be a non-empty 2-D matrix")
        basis = _fourier_basis(W.shape[1], self.num_coeffs)
        coeffs, _, _, _ = np.linalg.lstsq(basis, W.T, rcond=None)
        codes, lo, scale = _quantize(coeffs.T, self.bits)
        original = W.size * 8
        stored = codes.size * 2 + 16
        return {"shape": W.shape, "codes": codes, "lo": lo, "scale": scale}, original / max(stored, 1)

    def _decode(self, genes: dict) -> np.ndarray:
        rows, cols = genes["shape"]
        coeffs = _dequantize(genes["codes"], genes["lo"], genes["scale"])
        basis = _fourier_basis(cols, self.num_coeffs)
        return coeffs @ basis.T

    def put(self, name: str, weights: np.ndarray) -> float:
        """Encode and store a matrix; returns its compression ratio."""
        if not name:
            raise ValueError("name must be non-empty")
        genes, ratio = self._encode(weights)
        self._names.append(name)
        self._keys.append(self._embed(name))
        self._genes.append(genes)
        return ratio

    def get(self, name: str) -> np.ndarray:
        """Recall and decode a stored matrix by exact key."""
        return self._recall(self._embed(name), name)

    def get_noisy(self, name: str, noise: float = 0.05, seed: int = 1) -> np.ndarray:
        """Recall from a perturbed key wave."""
        if not 0.0 <= noise < 1.0:
            raise ValueError("noise must be in [0, 1)")
        rng = np.random.default_rng(seed + len(name))
        probe = self._embed(name) + noise * rng.standard_normal(self.dim)
        probe = probe / np.linalg.norm(probe)
        return self._recall(probe, name)

    def _recall(self, probe: np.ndarray, name: str) -> np.ndarray:
        if not self._keys:
            raise RecallMiss("store is empty")
        sims = np.stack(self._keys) @ probe
        best = int(np.argmax(sims))
        if float(sims[best]) < self.threshold:
            raise RecallMiss("no model resonates above threshold")
        return self._decode(self._genes[best])

    def list_models(self) -> list:
        return sorted(self._names)

    def __len__(self) -> int:
        return len(self._names)


class ActivationHelixStore(HybridStore):
    """HELIX storage specialised for activation matrices.

    put_activations() stores a (samples x features) activation block
    truncated to its top ``rank`` singular components and reports the
    retained variance. get_activations() recalls by exact name and
    get_noisy() recalls through a perturbed key wave, like the base store.
    """

    def put_activations(self, layer: str, activations: np.ndarray, rank: int = 8):
        """SVD-truncate activations, store them, return (ratio, retained)."""
        acts = np.asarray(activations, dtype=np.float64)
        if acts.ndim != 2 or acts.size == 0:
            raise ValueError("activations must be a non-empty 2-D matrix")
        if rank < 1:
            raise ValueError("rank must be >= 1")
        u, s, vt = np.linalg.svd(acts, full_matrices=False)
        rank = min(rank, s.size)
        total = float((s ** 2).sum())
        retained = float((s[:rank] ** 2).sum() / total) if total > 0 else 1.0
        approx = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
        ratio = self.put(layer, approx)
        self._retained = getattr(self, "_retained", {})
        self._retained[layer] = retained
        return ratio, retained

    def get_activations(self, layer: str) -> np.ndarray:
        """Recall a stored activation block by exact name."""
        return self.get(layer)

    def retained_variance(self, layer: str) -> float:
        """Retained variance reported when the layer was stored."""
    try:
        return self._retained[layer]
    except (AttributeError, KeyError) as exc:
        raise KeyError("layer not stored: {}".format(layer)) from exc

    def capture_from_model(self, model, inputs, layer_pattern: str = ""):
        """Capture activations from a PyTorch model via forward hooks.

        Requires torch (imported lazily). Stores every Linear/Conv2d
        output whose module name contains layer_pattern. Returns the
        list of stored layer names.
        """
        try:
            import torch  # noqa: F401
            import torch.nn as _nn
        except ImportError as exc:  # pragma: no cover
            raise ImportError("capture_from_model requires torch") from exc
        captured = {}

        def hook(_module, _inp, out):
            captured[len(captured)] = out.detach().float().cpu().numpy()
            return out

        handles = []
        for name, module in model.named_modules():
            if isinstance(module, (_nn.Linear, _nn.Conv2d)) and layer_pattern in name:
                handles.append(module.register_forward_hook(hook))
        try:
            import torch as _torch
            with _torch.no_grad():
                if isinstance(inputs, dict):
                    model(**inputs)
                else:
                    model(inputs)
        finally:
            for h in handles:
                h.remove()
        stored = []
        for i, act in sorted(captured.items()):
            self.put_activations("layer_{}".format(i), act)
            stored.append("layer_{}".format(i))
        return stored

    def list_layers(self) -> list:
        """Sorted names of stored activation blocks."""
        return self.list_models()
