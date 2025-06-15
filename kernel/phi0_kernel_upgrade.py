import numpy as np
from scipy.linalg import expm
from scipy.ndimage import gaussian_gradient_magnitude, gaussian_laplace
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def compute_radial_curvature_index(elevation: np.ndarray, radius: int) -> np.ndarray:
    """Compute curvature using a Gaussian Laplacian."""
    lap = gaussian_laplace(elevation, sigma=radius / 2.0)
    lap_norm = (lap - lap.min()) / (lap.max() - lap.min() + 1e-8)
    return lap_norm


def extract_8d_psi0_features(elevation: np.ndarray, resolution_m: float = 0.5) -> np.ndarray:
    """Return 8D psi0 feature tensor with Radial Curvature Index as feature[6]."""
    foundation_radius_px = int(4 / resolution_m)
    features = np.zeros(elevation.shape + (8,), dtype=float)

    # Feature 1: Radial Height Prominence (placeholder using gradient magnitude)
    features[..., 0] = gaussian_gradient_magnitude(elevation, sigma=1)
    # Feature 2: Circular Symmetry Score (placeholder using Laplacian)
    features[..., 1] = np.abs(gaussian_laplace(elevation, sigma=foundation_radius_px))
    # Feature 3: Radial Gradient Consistency (placeholder)
    features[..., 2] = gaussian_gradient_magnitude(elevation, sigma=2)
    # Feature 4: Ring Edge Sharpness (placeholder)
    features[..., 3] = np.clip(np.gradient(elevation)[0], 0, None)
    # Feature 5: Circular Hough Response (placeholder with zeros)
    features[..., 4] = 0
    # Feature 6: Foundation Planarity (placeholder using negative Laplacian)
    features[..., 5] = -gaussian_laplace(elevation, sigma=1)
    # Feature 7: Radial Curvature Index (new)
    features[..., 6] = compute_radial_curvature_index(elevation, foundation_radius_px)
    # Feature 8: Geometric Coherence (placeholder)
    features[..., 7] = gaussian_gradient_magnitude(elevation, sigma=foundation_radius_px)

    # Normalise features
    for i in range(8):
        f = features[..., i]
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        features[..., i] = f
    return features


def generate_g2_generators(dim: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    generators = []
    for _ in range(7):
        A = rng.standard_normal((dim, dim))
        G = A - A.T  # make antisymmetric
        generators.append(G)
    return generators


def interpolate_g2_kernels(phi_kernel: np.ndarray, g2_generators, num_variants: int = 7, alpha: float = 0.15):
    kernel_variants = []
    for i in range(num_variants):
        G = g2_generators[i % len(g2_generators)]
        R = expm(alpha * G)
        phi_rotated = R @ phi_kernel @ R.T
        kernel_variants.append(phi_rotated)
    return kernel_variants


def evaluate_kernel(kernel: np.ndarray, X: np.ndarray, y: np.ndarray):
    flat_kernel = kernel.reshape(-1)
    flat_data = X.reshape(len(X), -1)
    scores = flat_data @ flat_kernel
    pos_scores = scores[y == 1]
    neg_scores = scores[y == 0]
    gap = pos_scores.mean() - neg_scores.mean()
    auc = roc_auc_score(y, scores)
    return scores, gap, auc


def evaluate_kernel_variants(kernels, X_val, y_val):
    metrics = []
    for kern in kernels:
        scores, gap, auc = evaluate_kernel(kern, X_val, y_val)
        metrics.append({'kernel': kern, 'scores': scores, 'gap': gap, 'auc': auc})
    return metrics


def plot_validation(metrics):
    aucs = [m['auc'] for m in metrics]
    plt.figure(figsize=(6,4))
    plt.bar(range(len(aucs)), aucs)
    plt.xlabel('Kernel Variant')
    plt.ylabel('ROC AUC')
    plt.tight_layout()
    plt.savefig('kernel_validation_auc.png')


def main():
    rng = np.random.default_rng(42)
    # Dummy validation dataset
    X_pos = rng.standard_normal((50, 8, 8)) + 0.5
    X_neg = rng.standard_normal((50, 8, 8))
    X_val = np.concatenate([X_pos, X_neg])
    y_val = np.array([1]*len(X_pos) + [0]*len(X_neg))

    phi0_base = rng.standard_normal((8,8))
    g2_generators = generate_g2_generators()
    kernels = interpolate_g2_kernels(phi0_base, g2_generators)
    metrics = evaluate_kernel_variants(kernels, X_val, y_val)
    plot_validation(metrics)
    best_idx = int(np.argmax([m['auc'] for m in metrics]))
    best_kernel = metrics[best_idx]['kernel']
    np.save('best_phi0_kernel.npy', best_kernel)
    print(f"Best kernel variant: {best_idx} | AUC={metrics[best_idx]['auc']:.3f}")


if __name__ == "__main__":
    main()
