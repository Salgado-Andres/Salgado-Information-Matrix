"""Run a basic φ⁰ simulation."""
import numpy as np
from src.features import extract_psi0_features
from src.phi0 import phi0_score


def main() -> None:
    rng = np.random.default_rng(42)
    image = rng.random((32, 32))
    features = extract_psi0_features(image)
    score = phi0_score(features)
    print(f"φ⁰ score: {score:.4f}")


if __name__ == "__main__":
    main()
