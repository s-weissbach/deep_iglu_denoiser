from deep_iglu_denoiser.model.unet import UNet
import deep_iglu_denoiser.utils.normalization as normalization
from deep_iglu_denoiser.utils.write_file import write_file
from deep_iglu_denoiser.utils.open_file import open_file
from deep_iglu_denoiser.utils.convert import float_to_uint
import torch
import numpy as np


class ModelWrapper:
    """
    Wrapper class for a U-Net model used for image denoising.

    Parameters:
    - weights (str): Path to the pre-trained weights file.
    - n_pre (int): Number of frames to use before the target frame.
    - n_post (int): Number of frames to use after the target frame.
    """

    def __init__(self, weights: str, batch_size: int, cpu: bool) -> None:
        """
        Initialize the ModelWrapper.

        Initializes the U-Net model, loads pre-trained weights, and sets up device (GPU or CPU).

        Parameters:
        - weights (str): Path to the pre-trained weights file.
        - n_pre (int): Number of frames to use before the target frame.
        - n_post (int): Number of frames to use after the target frame.
        """
        # initalize model
        self.batch_size = batch_size
        # check for GPU, use CPU otherwise
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if flag cpu is set, use cpu regardless of available GPU
        if cpu:
            self.device = "cpu"
        self.model = UNet(1)
        self.load_weights(weights)
        self.model.to(self.device)
        # initalize image
        self.denoised_img = np.empty((0, 0, 0))
        self.img = np.empty((0, 0, 0))
        self.img_height = -1
        self.img_width = -1
        self.img_mean = np.empty((0, 0))
        self.img_std = np.empty((0, 0))

    def load_weights(self, weights: str) -> None:
        """
        Load pre-trained weights into the U-Net model.

        Parameters:
        - weights (str): Path to the pre-trained weights file.
        """
        if self.device == "cpu":
            self.model.load_state_dict(
                torch.load(weights, map_location=torch.device("cpu"))
            )
        else:
            self.model.load_state_dict(torch.load(weights))
        self.model.eval()

    def load_and_normalize_img(self, img_path: str) -> None:
        """
        Load an image from the specified path, perform normalization, and store information about the image.

        Parameters:
        - img_path (str): Path to the image file.
        """
        self.img: np.ndarray = open_file(img_path)
        _, self.img_height, self.img_width = self.img.shape
        # compute mean and std along z-axis
        self.img_mean: np.ndarray = np.mean(self.img, axis=0)
        self.img_std: np.ndarray = np.std(self.img, axis=0)
        # normalization
        self.img: np.ndarray = normalization.z_norm(
            self.img, self.img_mean, self.img_std
        )

    def get_prediction_frames(self, from_frame: int) -> torch.Tensor:
        """
        Extract frames around the target frame for making predictions.

        Parameters:
        - target (int): Index of the target frame.

        Returns:
        - torch.Tensor: Input tensor for the U-Net model.
        """
        to_frame = min(len(self.img), from_frame + self.batch_size)
        # extract frames
        X = self.img[from_frame:to_frame]
        # reshape to batch size 1
        X = X.reshape(
            min(self.batch_size, to_frame - from_frame),
            1,
            self.img_height,
            self.img_width,
        )
        return torch.tensor(X, dtype=torch.float)

    def denoise_img(self, img_path: str) -> None:
        """
        Denoise an image sequence using the U-Net model.

        Parameters:
        - img_path (str): Path to the image sequence file.

        Returns:
        - np.ndarray: Denoised image sequence.
        """
        denoised_image_sequence = []
        self.load_and_normalize_img(img_path)
        for from_frame in range(0, self.img.shape[0], self.batch_size):
            X = self.get_prediction_frames(from_frame).to(self.device)
            y_pred = np.array(self.model(X).detach().to("cpu"))
            for denoised_frame in y_pred:
                denoised_image_sequence.append(
                    denoised_frame.reshape(self.img_height, self.img_width)
                )
        self.denoised_img = normalization.reverse_z_norm(
            np.array(denoised_image_sequence), self.img_mean, self.img_std
        )
        # tiff format is based on uint16 -> cast
        self.denoised_img = float_to_uint(self.denoised_img)

    def write_denoised_img(self, outpath: str) -> None:
        if self.denoised_img.shape[0] == 0:
            raise AssertionError(
                f"Before writing a denoised image, first denoise image. Use <ModelWrapper>.denoise_img(<path/to/input_image>)."
            )
        write_file(self.denoised_img, outpath)
