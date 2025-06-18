import numpy as np


def extract_psi0_features(image: np.ndarray) -> np.ndarray:
    """Generate a simple 8-channel feature tensor.

    Parameters
    ----------
    image : np.ndarray
        2D input array representing an image.

    Returns
    -------
    np.ndarray
        Feature tensor of shape ``(H, W, 8)``.
    """
    if image.ndim != 2:
        raise ValueError("image must be 2D")

    h, w = image.shape
    features = np.zeros((h, w, 8), dtype=np.float32)

    gx, gy = np.gradient(image)

    features[..., 0] = image
    features[..., 1] = image ** 2
    features[..., 2] = gx
    features[..., 3] = gy
    features[..., 4] = np.sqrt(gx ** 2 + gy ** 2)
    features[..., 5] = gx * gy
    features[..., 6] = (image - image.mean()) / (image.std() + 1e-6)
    features[..., 7] = (image > image.mean()).astype(np.float32)

    return features
