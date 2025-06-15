"""
FULL φ⁰ VALIDATION PIPELINE

This script auto-loads training and validation patch data, rebuilds φ⁰ using real features,
runs validation across the dataset, and outputs coherence score plots + ROC/AUC.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from phi0_kernel_upgrade import (
    extract_8d_psi0_features,
    generate_g2_generators,
    interpolate_g2_kernels,
    evaluate_kernel_variants,
)

training_patches = None
validation_patches = None


def load_real_patch_data():
    """Load training and validation patches from disk if not already loaded."""
    global training_patches, validation_patches

    if training_patches is None:
        try:
            with open("training_patches.pkl", "rb") as f:
                training_patches = pickle.load(f)
            print("✅ Loaded training_patches from file.")
        except Exception as e:
            raise RuntimeError("❌ training_patches missing and not found on disk.") from e

    if validation_patches is None:
        try:
            with open("validation_patches.pkl", "rb") as f:
                validation_patches = pickle.load(f)
            print("✅ Loaded validation_patches from file.")
        except Exception as e:
            raise RuntimeError("❌ validation_patches missing and not found on disk.") from e


def _cov_matrix_from_patch(patch):
    """Return 8x8 covariance matrix from a patch dict containing 'elevation'."""
    elev = patch["elevation"] if isinstance(patch, dict) else np.array(patch)
    psi = extract_8d_psi0_features(np.array(elev))
    flat = psi.reshape(-1, 8)
    return np.cov(flat, rowvar=False)


def run_validation_with_real_data():
    """Construct φ⁰ kernel from training data and validate on held-out patches."""
    g2_generators = generate_g2_generators()

    # build kernel from training data
    covmats = [_cov_matrix_from_patch(p) for p in training_patches]
    base_kernel = np.mean(covmats, axis=0)
    kernels = interpolate_g2_kernels(base_kernel, g2_generators)

    # prepare validation data
    X_val = []
    y_val = []
    for p in validation_patches:
        X_val.append(_cov_matrix_from_patch(p))
        y_val.append(1 if (isinstance(p, dict) and p.get("is_positive", True)) else 0)
    X_val = np.stack(X_val)
    y_val = np.array(y_val)

    metrics = evaluate_kernel_variants(kernels, X_val, y_val)

    # compute optimal threshold using best kernel
    best_idx = int(np.argmax([m["auc"] for m in metrics]))
    best_scores = metrics[best_idx]["scores"]
    pos_mean = best_scores[y_val == 1].mean()
    neg_mean = best_scores[y_val == 0].mean()
    threshold = (pos_mean + neg_mean) / 2

    return {
        "metrics": metrics,
        "optimal_threshold": threshold,
        "roc_auc": metrics[best_idx]["auc"],
        "best_kernel": metrics[best_idx]["kernel"],
    }


def plot_validation_summary(metrics):
    """Plot bar chart of AUC for kernel variants."""
    plt.figure(figsize=(6, 4))
    aucs = [m["auc"] for m in metrics]
    plt.bar(range(len(aucs)), aucs)
    plt.xlabel("Kernel Variant")
    plt.ylabel("ROC AUC")
    plt.tight_layout()
    plt.savefig("phi0_validation_summary.png")
    print("📁 Saved plot: phi0_validation_summary.png")


def full_phi0_validation():
    print("\n🚀 Starting φ⁰ validation from real data...")
    load_real_patch_data()

    results = run_validation_with_real_data()

    plot_validation_summary(results["metrics"])

    np.save("best_phi0_kernel.npy", results["best_kernel"])

    print("\n🌿 Validation complete.")
    print(f"   ➤ Optimal Threshold: {results['optimal_threshold']:.4f}")
    print(f"   ➤ ROC AUC:           {results['roc_auc']:.4f}")


if __name__ == "__main__":
    try:
        full_phi0_validation()
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
