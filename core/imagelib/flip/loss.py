import numpy as np
from core.imagelib.flip.flip import compute_flip


def flip_loss(reference, test):
    """Expects image, in CHW format, as float32 in BGR colorspace"""

    # reverse channels, BGR -> RGB
    reference = np.flip(reference, axis=-3)
    test = np.flip(test, axis=-3)

    # Set viewing conditions
    monitor_distance = 0.7
    monitor_width = 0.7
    monitor_resolution_x = 3840

    # Compute number of pixels per degree of visual angle
    pixels_per_degree = monitor_distance * (monitor_resolution_x / monitor_width) * (np.pi / 180)

    result = compute_flip(reference, test, pixels_per_degree)
    return 1 - np.mean(result, axis=[-1, -2])
