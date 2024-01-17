import numpy as np
from scipy.ndimage import uniform_filter1d


def z_norm(img: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Pixelwise z-scaling for the image. z = (x-µ)/σ

    Parameters:
    - img: Input image.
    - mean: Mean matrix for scaling
    - std: Standard deviation matrix for scaling.

    Returns:
    - z-scaled image.
    """
    return np.divide(np.subtract(img, mean), std)

def moving_std(arr:np.ndarray[np.float64], start:int, end:int) -> np.ndarray[np.float64]:
    """
    Calculate the moving standard deviation for a specified range of frames.

    Parameters:
    - parameters: Tuple containing the start and end indices.

    Returns:
    - np.ndarray: Moving standard deviation values.
    """
    std = np.std(arr[start:end], axis=0)
    return std

def rolling_window_z_norm(img: np.ndarray, window_size: int):
    """
    Apply rolling window z-scaling to an image sequence.

    Parameters:
    - img: Input image sequence.
    - window_size: Size of the rolling window.
    - n_threads: Number of threads for parallel processing (default is 1).

    Returns:
    - np.ndarray: Z-scaled image sequence.
    """
    before = window_size // 2
    after = window_size - before
    mean = uniform_filter1d(img,window_size,axis=0,mode='constant').astype(np.float16)
    std = np.zeros_like(img, dtype=np.float16)
    for idx in range(img.shape[0]):
        start = max(0, idx - before)
        end = min(img.shape[0] - 1, idx + after)
        std[idx,:,:] += (moving_std(img, start, end))
    normed_img = np.divide(np.subtract(img, mean), std).astype(np.float16)
    return normed_img


def reverse_z_norm(img: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Reverse z-scaling for the image. x = z*σ+µ

    Parameters:
    - img: Input image.
    - mean: Mean matrix for scaling.
    - std: Standard deviation matrix for scaling.

    Returns:
    - reversed z-scored image.
    """
    return np.add(np.multiply(img, std), mean)
