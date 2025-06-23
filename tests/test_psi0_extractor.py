import numpy as np
from src.psi0_extractor import PSI0FeatureExtractor


def test_tile_feature_vector_length():
    dsm = np.random.rand(32, 32)
    extractor = PSI0FeatureExtractor()
    feats = extractor.extract_tile_features(dsm)
    assert feats.shape == (5,)
    assert np.all((feats >= 0) & (feats <= 1))

def test_individual_features():
    dsm = np.random.rand(16, 16)
    extractor = PSI0FeatureExtractor()
    funcs = [
        extractor.radial_variance_normalized,
        extractor.hough_entropy,
        extractor.fractal_surface_dimension,
        extractor.spectral_torsion_score,
        extractor.aspect_convergence_index,
    ]
    for f in funcs:
        val = f(dsm)
        assert 0.0 <= val <= 1.0
