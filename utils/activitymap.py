import numpy as np
from scipy.ndimage import uniform_filter


def transform_frame(
    frame: np.ndarray,
    kernelsize: int,
    h_activitymap: int,
    w_activitymap: int,
    roi_size: int,
) -> list[float]:
    activity_map_frame = []
    mean_frame = uniform_filter(frame, roi_size, mode="constant")
    for y in range(h_activitymap):
        start_y = y * kernelsize
        stop_y = (y + 1) * kernelsize
        row = []
        for x in range(w_activitymap):
            start_x = x * kernelsize
            stop_x = (x + 1) * kernelsize

            row.append(np.max(mean_frame[start_y:stop_y, start_x:stop_x]))
        activity_map_frame.append(row)
    return activity_map_frame


def compute_activitymap(img: np.ndarray, kernelsize: int, roi_size: int) -> np.ndarray:
    """
    Compute the activity map for a sequence of image frames by applying the
    transform_frame function.

    Parameters:
    - img (np.ndarray): Input image sequence as a 3D NumPy array.
    - kernelsize: Size of the kernel used for patch extraction (image size on that is trained).
    - roi_size: Size of the sliding window (Region of Interest).

    Returns:
    - np.ndarray: Activity map for the input image sequence.
    """
    h_activitymap = img.shape[1] // kernelsize
    w_activitymap = img.shape[2] // kernelsize
    activitymap = []
    for frame_idx in range(img.shape[0]):
        frame = img[frame_idx]
        activitymap.append(
            transform_frame(frame, kernelsize, h_activitymap, w_activitymap, roi_size)
        )
    return np.array(activitymap)


def get_frames_position(
    img: np.ndarray,
    min_z_score: float,
    before: int,
    after: int,
    kernelsize: int = 32,
    roi_size: int = 4,
    foreground_background_split: float = 0.5,
) -> list[list[int]]:
    """
    Identify positions of frames based on the computed activity map and a minimum Z-score threshold.

    Parameters:
    - img (np.ndarray): Input image sequence as a 3D NumPy array.
    - min_z_score (float): Minimum Z-score threshold for identifying frames.
    - kernelsize (int): Size of the kernel used for patch extraction.
    - roi_size (int): Size of the sliding window (Region of Interest).
    - foreground_background_split (float): Split ratio between foreground and background.

    Returns:
    - list[list[int]]: List of frame positions, each represented as [frame_index, y_position, x_position].
    """
    frames_w_pos = []
    activitymap = compute_activitymap(img, kernelsize, roi_size)
    above_z = np.argwhere(activitymap > min_z_score)
    for example in above_z:
        frame, y, x = example
        frames_w_pos.append([int(frame), int(y * kernelsize), int(x * kernelsize)])
        for i in range(1, before + 1):
            example_to_add = [int(frame) - i, int(y * kernelsize), int(x * kernelsize)]
            if example_to_add in frames_w_pos:
                continue
            frames_w_pos.append(example_to_add)
        for i in range(1, after + 1):
            example_to_add = [int(frame) + i, int(y * kernelsize), int(x * kernelsize)]
            if example_to_add in frames_w_pos:
                continue
            frames_w_pos.append(example_to_add)
    bg_images_to_select = (1 / foreground_background_split - 1) * len(frames_w_pos)
    below_z = np.argwhere(activitymap <= min_z_score)
    np.random.shuffle(below_z)
    for i, example in enumerate(below_z):
        if i > bg_images_to_select:
            break
        frame, y, x = example
        frames_w_pos.append([int(frame), int(y * kernelsize), int(x * kernelsize)])
    return frames_w_pos
