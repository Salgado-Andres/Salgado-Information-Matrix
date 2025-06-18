import numpy as np
from src.features import extract_psi0_features


def test_extract_psi0_features_shape():
    img = np.zeros((10, 20))
    feats = extract_psi0_features(img)
    assert feats.shape == (10, 20, 8)
