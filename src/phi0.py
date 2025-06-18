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
