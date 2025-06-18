import numpy as np
from src.features import extract_psi0_features
from src.phi0 import phi0_score


def test_phi0_score_scalar():
    rng = np.random.default_rng(0)
    img = rng.random((5, 5))
    features = extract_psi0_features(img)
    score = phi0_score(features)
    assert isinstance(score, float)
