import numpy as np

phi0 = np.load("best_phi0_kernel.npy")
print("🧠 Loaded φ⁰ Kernel Matrix (8x8):\n")
np.set_printoptions(precision=4, suppress=True)
print(phi0)

# Optional: print summary stats
print("\n📊 Kernel stats:")
print(f"Mean: {np.mean(phi0):.4f} | Std: {np.std(phi0):.4f} | Max: {np.max(phi0):.4f} | Min: {np.min(phi0):.4f}")
