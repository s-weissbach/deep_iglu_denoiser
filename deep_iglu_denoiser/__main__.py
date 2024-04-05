import argparse
from alive_progress import alive_bar
import h5py
from scipy.ndimage import uniform_filter
import numpy as np
import yaml

from deep_iglu_denoiser.utils.trainfiles import TrainFiles
from deep_iglu_denoiser.utils.dataloader import DataLoader
from deep_iglu_denoiser.model.unet import UNet
from deep_iglu_denoiser.model.train import train
from deep_iglu_denoiser.model.denoise import inference

def main():
    parser = argparse.ArgumentParser(description='iGlu Denoiser')
    subparsers = parser.add_subparsers(dest='mode')

    pre_training_p = subparsers.add_parser('prepare_training')
    pre_training_p.add_argument(
        "--path", "-p", required=True, help="Path to folder containing images"
    )
    pre_training_p.add_argument(
        "--fileendings",
        "-f",
        required=True,
        nargs="+",
        help="List of file endings to consider",
    )
    # Optional arguments
    pre_training_p.add_argument(
        "--crop_size",
        "-c",
        type=int,
        default=32,
        help="Crop size used during training (default: 32)",
    )
    pre_training_p.add_argument(
        "--roi_size",
        type=int,
        default=4,
        help="Expected ROI size; assumes for detection square of (roi_size x roi_size) (default: 8)",
    )
    pre_training_p.add_argument(
        "--h5",
        required=True,
        help="Path to outputpath of the h5 file that will be created",
    )
    pre_training_p.add_argument(
        "--min_z_score",
        "-z",
        type=float,
        default=2.0,
        help="Minimum Z score to be considered active patch (default: 2)",
    )
    pre_training_p.add_argument(
        "--window_size",
        "-w",
        type=int,
        default=50,
        help="Number of frames used for rolling window z-normalization (default: 50)",
    )
    pre_training_p.add_argument(
        "--fgsplit",
        "-s",
        type=float,
        default=0.5,
        help="Foreground to background split (default: 0.5)",
    )
    pre_training_p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing h5 file. If false, data will be appended. (default: False)",
    )
    pre_training_p.add_argument(
        "--memory_optimized",
        action="store_true",
        help="Utilize optimized memory mode. Trades speed for lower memory usage",
    )
    # Training
    train_p = subparsers.add_parser('train')
    train_p.add_argument(
        "--trainconfigpath", "-p", required=True, help="Path to train config YAML file"
    )
    # Filter
    filter_p = subparsers.add_parser('filter')
    filter_p.add_argument(
        "--h5", type=str, help="Path to the input H5 file", required=True
    )
    filter_p.add_argument(
        "--output_h5", "-o", type=str, help="Path to the output H5 file", required=True
    )
    filter_p.add_argument(
        "--min_z", "-z", type=float, help="Minimum Z value", required=True
    )
    filter_p.add_argument(
        "--roi_size",
        "-r",
        type=int,
        help="Size of the Region of Interest (ROI)",
        required=True,
    )
    # Denoise / Inference
    denoise_p = subparsers.add_parser('denoise')
    denoise_p.add_argument(
        "--path", "-p", type=str, required=True, help="Specify the path."
    )
    denoise_p.add_argument(
        "--modelpath", "-m", type=str, required=True, help="Path to modelweights."
    )
    denoise_p.add_argument(
        "--directory_mode", "-d", action="store_true", help="Enable directory mode."
    )
    denoise_p.add_argument(
        "--outputpath", "-o", type=str, required=True, help="Specify the output path."
    )
    denoise_p.add_argument(
        "--batchsize",
        "-b",
        type=int,
        default=1,
        help="Number of frames that are predicted at once.",
    )
    denoise_p.add_argument("--cpu", action="store_true", help="Force CPU and not use GPU.")
    
    args = parser.parse_args()
    if args.mode == 'prepare_training':
        trainfiles = TrainFiles(
            fileendings=args.fileendings,
            min_z_score=args.min_z_score,
            crop_size=args.crop_size,
            roi_size=args.roi_size,
            output_h5_file=args.h5,
            window_size=args.window_size,
            foreground_background_split=args.fg_split,
            overwrite=args.overwrite,
        )
        # gather train data
        trainfiles.files_to_traindata(
            directory=args.folder_path,
            memory_optimized=args.memory_optimized,
        )
    # training
    elif args.mode == 'train':
        trainconfigpath = args.trainconfigpath
        # parse train config file
        with open(trainconfigpath, "r") as f:
            trainconfig = yaml.safe_load(f)
        modelpath = trainconfig["modelpath"]
        h5 = trainconfig["train_h5"]
        batch_size = trainconfig["batch_size"]
        learning_rate = trainconfig["learning_rate"]
        num_epochs = trainconfig["num_epochs"]
        noise_center = trainconfig["noise_center"]
        noise_scale = trainconfig["noise_scale"]
        dataloader = DataLoader(h5, batch_size, noise_center, noise_scale)
        model = UNet(1)
        train(model, dataloader, num_epochs, learning_rate, modelpath)
    # filter
    elif args.mode == 'filter':
        # input
        f_in = h5py.File(args.h5, "r")
        # "/mnt/nvme2/iGlu_train_data/iglu_train_data_cropsize32_roisize4_stim_z2_filtered.h5"
        f_out = h5py.File(args.output_h5, "w")
        idx = 0
        num_samples = f_in.__len__()
        with alive_bar(num_samples) as bar:
            for i in range(num_samples):
                frame = np.array(f_in.get(str(i)))
                mean_frame = uniform_filter(frame, args.roi_size, mode="constant")
                if np.any(mean_frame > args.min_z):
                    f_out.create_dataset(str(idx), data=frame)
                    idx += 1
                    if idx % 1_000 == 0:
                        print(f"Wrote {idx} files to the filtered h5-file.")
                bar()
        f_out.close()
        f_in.close()
        print(f"Kept {idx} of {num_samples} examples.")
    # denoising / inference
    elif args.mode == 'denoise':
        inference(
            args.path,
            args.modelpath,
            args.directory_mode,
            args.outputpath,
            args.batchsize,
            args.cpu,
        )

if __name__ == '__main__':
    main()