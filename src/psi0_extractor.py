import numpy as np
from skimage.transform import hough_circle
from skimage.feature import canny


class PSI0FeatureExtractor:
    """Extract ψ⁰ features from LiDAR DSM tiles."""

    def __init__(self, radial_bins: int = 8):
        self.radial_bins = radial_bins

    # ------------------------------------------------------------------
    @staticmethod
    def radial_variance_normalized(dsm: np.ndarray, radial_bins: int = 8) -> float:
        """Normalized variance of elevation in concentric radial bins."""
        h, w = dsm.shape
        cy, cx = h / 2.0, w / 2.0
        y, x = np.indices(dsm.shape)
        r = np.sqrt((y - cy)**2 + (x - cx)**2)
        max_r = r.max()
        bins = np.linspace(0, max_r, radial_bins + 1)
        variances = []
        for i in range(len(bins) - 1):
            mask = (r >= bins[i]) & (r < bins[i + 1])
            if np.any(mask):
                variances.append(float(np.var(dsm[mask])))
        if not variances:
            return 0.0
        radial_var = np.var(variances)
        global_var = np.var(dsm) + 1e-6
        return float(np.clip(radial_var / global_var, 0.0, 1.0))

    # ------------------------------------------------------------------
    @staticmethod
    def hough_entropy(dsm: np.ndarray) -> float:
        """Entropy of the Hough Circle accumulator to detect multi-ring patterns."""
        edges = canny(dsm.astype(float), sigma=1.0)
        h, w = dsm.shape
        radii = np.arange(3, max(4, min(h, w) // 2), 2)
        if len(radii) == 0:
            return 0.0
        accumulator = hough_circle(edges, radii)
        acc_sum = accumulator.sum(axis=0)
        hist, _ = np.histogram(acc_sum, bins=20, range=(acc_sum.min(), acc_sum.max()), density=True)
        hist = hist[hist > 0]
        if hist.size == 0:
            return 0.0
        entropy = -np.sum(hist * np.log(hist))
        max_entropy = np.log(len(hist))
        return float(np.clip(entropy / (max_entropy + 1e-12), 0.0, 1.0))

    # ------------------------------------------------------------------
    @staticmethod
    def fractal_surface_dimension(dsm: np.ndarray) -> float:
        """Estimate fractal dimension using box-counting."""
        z = dsm - np.min(dsm)
        max_size = min(dsm.shape)
        sizes = [2**i for i in range(1, int(np.log2(max_size)))]
        counts = []
        for size in sizes:
            S = np.add.reduceat(np.add.reduceat(z, np.arange(0, z.shape[0], size), axis=0),
                                np.arange(0, z.shape[1], size), axis=1)
            counts.append(np.sum(S > 0))
        if len(counts) < 2:
            return 0.0
        coeffs = np.polyfit(np.log(1 / np.array(sizes)), np.log(counts), 1)
        dim = coeffs[0]
        dim = np.clip((dim - 2.0) / 1.0, 0.0, 1.0)
        return float(dim)

    # ------------------------------------------------------------------
    @staticmethod
    def spectral_torsion_score(dsm: np.ndarray) -> float:
        """Measure torsional residuals via spectral gradient norms."""
        fft = np.fft.fft2(dsm)
        grad_y, grad_x = np.gradient(np.abs(fft))
        grad_norm = np.sqrt(grad_x**2 + grad_y**2)
        score = np.mean(grad_norm)
        normalization = np.mean(np.abs(fft)) + 1e-6
        return float(np.clip(score / normalization, 0.0, 1.0))

    # ------------------------------------------------------------------
    @staticmethod
    def aspect_convergence_index(dsm: np.ndarray, radial_bins: int = 8) -> float:
        """Standard deviation of aspect vectors in radial segments."""
        gy, gx = np.gradient(dsm)
        aspect = np.arctan2(gy, gx)
        h, w = dsm.shape
        cy, cx = h / 2.0, w / 2.0
        y, x = np.indices(dsm.shape)
        r = np.sqrt((y - cy)**2 + (x - cx)**2)
        max_r = r.max()
        bins = np.linspace(0, max_r, radial_bins + 1)
        vectors = []
        for i in range(len(bins) - 1):
            mask = (r >= bins[i]) & (r < bins[i + 1])
            if np.any(mask):
                vec = np.mean(np.exp(1j * aspect[mask]))
                vectors.append(vec)
        if len(vectors) < 2:
            return 0.0
        angles = np.angle(vectors)
        std = np.std(angles)
        return float(np.clip(1.0 - std / np.pi, 0.0, 1.0))

    # ------------------------------------------------------------------
    def extract_tile_features(self, dsm: np.ndarray) -> np.ndarray:
        """Extract all ψ⁰ features for a DSM tile."""
        return np.array([
            self.radial_variance_normalized(dsm, self.radial_bins),
            self.hough_entropy(dsm),
            self.fractal_surface_dimension(dsm),
            self.spectral_torsion_score(dsm),
            self.aspect_convergence_index(dsm, self.radial_bins),
        ], dtype=float)

    # ------------------------------------------------------------------
    def extract_point_features(self, dsm: np.ndarray) -> np.ndarray:
        """Alias for `extract_tile_features` for API compatibility."""
        return self.extract_tile_features(dsm)
