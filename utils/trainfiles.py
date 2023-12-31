import os
import tifffile
import yaml
import numpy as np
import utils.normalization as normalization
from utils.activitymap import get_frames_position


class TrainFiles:
    def __init__(self, train_yaml_path: str, overwrite: bool = False) -> None:
        self.train_yaml_path = train_yaml_path
        self.overwrite = overwrite
        self.file_dict = {}
        if os.path.exists(self.train_yaml_path):
            self.open_json()

    def open_json(self) -> None:
        if os.path.exists(self.train_yaml_path):
            with open(self.train_yaml_path, "r") as f:
                self.file_dict = yaml.safe_load(f)
        else:
            print(f"YAML path not found: {self.train_yaml_path}")

    def write_yaml(self) -> None:
        with open(self.train_yaml_path, "w") as f:
            yaml.dump(self.file_dict, f)

    def find_files(
        self,
        directory: str,
        fileendings: list[str],
        min_z_score: float,
        kernel_size: int,
        window_size: int = 50,
    ):
        self.file_dict = {}
        idx = 0
        for root, _, files in os.walk(directory):
            for file in files:
                if not any([file.endswith(ending) for ending in fileendings]):
                    continue
                filepath = os.path.join(root, file)
                tmp_file = tifffile.imread(filepath)
                mean = np.mean(tmp_file)
                std = np.std(tmp_file)
                # find train examples with activity
                tmp_file = normalization.rolling_window_z_norm(tmp_file, window_size)
                # will go through all frames and extract events that within a meaned kernel exceed the
                # min_z_score threshold
                # returns a list of events in the form [frame, y-coord, x-coord]
                frames_and_positions = get_frames_position(
                    tmp_file, min_z_score, kernel_size
                )
                self.file_dict[idx] = {
                    "filepath": filepath,
                    "shape": tmp_file.shape,
                    "mean": mean,
                    "std": std,
                    "frames_and_positions": frames_and_positions,
                }
                idx += 1
        if self.overwrite:
            self.write_yaml()
