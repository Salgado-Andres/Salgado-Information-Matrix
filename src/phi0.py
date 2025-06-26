import numpy as np


def phi0_score(features: np.ndarray) -> float:
    """Compute a simple φ⁰ score from the feature tensor.

    Parameters
    ----------
    features : np.ndarray
        Tensor of shape ``(H, W, C)``.

    Returns
    -------
    float
        Scalar score summarizing the features.
    """
    if features.ndim != 3:
        raise ValueError("features must be a 3D tensor")

    c = features.shape[-1]
    weights = np.arange(1, c + 1, dtype=np.float32)
    mean_features = features.reshape(-1, c).mean(axis=0)
    score = float(np.dot(mean_features, weights) / weights.sum())
    return score

from typing import Any, Dict, List



def collapse_field(psi_field: Dict[str, Any], depth: int = 1) -> Dict[str, Any]:
    """Collapse a ψ⁰ field into a structured symbolic attractor.

    Parameters
    ----------
    psi_field : dict
        Output from :func:`~src.psi0.generate_field`.
    depth : int
        Recursion depth controlling identity feedback.

    Returns
    -------
    dict
        Mapping with keys ``phi0``, ``feedback``, and ``candidates``.
    """
    field: List[str] = psi_field.get("field", [])
    if not field:
        return {"phi0": "", "feedback": "", "candidates": []}

    tokens = " ".join(field).split()
    unique_tokens = sorted(set(tokens))
    phi0 = " ".join(unique_tokens[:3])

    feedback = ""
    if depth > 1:
        feedback = f"Σ preserved through {depth} collapses"

    candidates = list(psi_field.get("candidates", []))
    if phi0:
        candidates.append(phi0[::-1])

    return {"phi0": phi0, "feedback": feedback, "candidates": candidates}

class Phi0:
    def __init__(self):
        pass

    def collapse(self, psi_out, depth=1):
        return collapse_field(psi_out, depth=depth)

