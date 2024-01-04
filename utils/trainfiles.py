import os
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
import utils.normalization as normalization
from utils.activitymap import get_frames_position
from utils.open_file import open_file


class TrainFiles:
    def __init__(self, train_csv_path: str, overwrite: bool = False) -> None:
        self.train_csv_path = train_csv_path
        self.overwrite = overwrite
        self.file_list = {}
        if os.path.exists(self.train_csv_path):
            self.open_csv()
        else:
            self.train_examples = pd.DataFrame(columns=['h5_idx', 'original_filepath', 'target_frame', 'y_pos', 'x_pos'])

    def open_csv(self) -> None:
        if os.path.exists(self.train_csv_path):
            self.train_examples = pd.read_csv(self.train_csv_path)
        else:
            print(f"CSV path not found: {self.train_csv_path}")
    
    def get_train_example(self, img: np.ndarray, target_frame: int, y_pos: int, x_pos: int, height: int, width: int, n_pre: int, n_post: int) -> np.ndarray:
        '''
        Extracts a single train example and crops it.
        '''
        example = img[target_frame - n_pre: target_frame + n_post + 1]
        return example[:,y_pos:y_pos+height,x_pos:x_pos+width]

    def files_to_traindata(
        self,
        directory: str,
        fileendings: list[str],
        min_z_score: float,
        kernel_size: int,
        output_h5_file: str,
        n_pre: int = 2,
        n_post: int = 2,
        window_size: int = 50,
        n_threads: int = 1,
        foreground_background_split: float = 0.1,
    ):
        '''
        Iterates through given directory and searches for all files having the specified fileendings. Opens these images and extracts active/inactive
        patches (size: kernel_size) from the files (above/below min_z_score) and writes a single h5 file for training. active: foreground; inactive: background are
        splitted according to foreground_background_split.
        A train example consists of n_pre and n_post frames around the target frame.
        '''
        train_example_list = []
        files_to_do = []
        for root, _, files in os.walk(directory):
            for file in files:
                if not any([file.endswith(ending) for ending in fileendings]):
                    continue
                files_to_do.append(os.path.join(root, file))
        if os.path.exists(output_h5_file) and not self.overwrite:
            print('Found existing h5-file. Will append.')
            hf = h5py.File(output_h5_file, 'w')
            idx = np.max([int(key) for key in hf.keys()]) + 1
        elif os.path.exists(output_h5_file) and self.overwrite:
            os.remove(output_h5_file)
            hf = h5py.File(output_h5_file, 'w')
            idx = 0
        else:
            hf = h5py.File(output_h5_file, 'w')
            idx = 0
        for filepath in tqdm(files_to_do, total=len(files_to_do)):
            tmp_file = open_file(filepath)
            mean = np.mean(tmp_file,axis=0)
            std = np.std(tmp_file,axis=0)
            # find train examples with activity
            tmp_file_rolling_normalization = normalization.rolling_window_z_norm(
                tmp_file, window_size, n_threads=n_threads
            )
            # will go through all frames and extract events that within a meaned kernel exceed the
            # min_z_score threshold
            # returns a list of events in the form [frame, y-coord, x-coord]
            frames_and_positions = get_frames_position(
                tmp_file_rolling_normalization, min_z_score, kernel_size, foreground_background_split
            )
            print(f'Found {len(frames_and_positions)} example(s) in file {filepath}')
            tmp_file = normalization.z_norm(tmp_file, mean, std)
            # create dict to be stored as h5 file
            for train_example,event in enumerate(frames_and_positions):
                target_frame, y_pos, x_pos = event
                train_example_list.append([str(train_example), filepath, target_frame, y_pos, x_pos])
                example = self.get_train_example(tmp_file,target_frame, y_pos, x_pos, kernel_size, kernel_size, n_pre, n_post)
                hf.create_dataset(str(idx), data=example)
                idx += 1
        hf.close()
        self.train_examples = pd.DataFrame(train_example_list,columns=['h5_idx', 'original_filepath', 'target_frame', 'y_pos', 'x_pos'])
        if self.overwrite:
            self.train_examples.to_csv(self.train_csv_path)
