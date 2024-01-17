import argparse
import os
from tqdm import tqdm
from model.modelwrapper import ModelWrapper


def main(
    path: str,
    modelpath: str,
    n_pre: int,
    n_post: int,
    directory_mode: str,
    outputpath: str,
) -> None:
    valid_fileendings = [".tif", ".tiff", ".stk", ".nd2"]
    # initalize model
    model = ModelWrapper(modelpath, n_pre, n_post)
    if not directory_mode:
        if not any([path.endswith(ending) for ending in valid_fileendings]):
            raise NotImplementedError(
                f"Fileending .{path.split('.')[-1]} is currently not implemented."
            )
        filelist = [path]
    else:
        filelist = []
        for folderpath, _, filenames in os.walk(path):
            for filename in filenames:
                if not any([filename.endswith(ending) for ending in valid_fileendings]):
                    continue
                filelist.append(os.path.join(folderpath, filename))
    for filepath in tqdm(filelist, total=len(filelist), desc="Image Denoising"):
        filename = os.path.splitext(os.path.basename(filepath))[0]
        outfilepath = os.path.join(outputpath, f"{filename}_denoised.tif")
        if os.path.exists(outfilepath):
            print(f"\nSkipped image ({os.path.basename(filepath)}) – already denoised.")
            continue
        model.denoise_img(filepath)
        model.write_denoised_img(outfilepath)
        print(f"\nSaved image ({os.path.basename(filepath)}) as: {outfilepath}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Custom Argument Parser")

    parser.add_argument(
        "--path", "-p", type=str, required=True, help="Specify the path."
    )
    parser.add_argument(
        "--modelpath", "-m", type=str, required=True, help="Path to modelweights."
    )
    parser.add_argument(
        "--n_pre", type=int, required=True, help="Frames before target frame."
    )
    parser.add_argument(
        "--n_post", type=int, required=True, help="Frames after target frame."
    )
    parser.add_argument(
        "--directory_mode", "-d", action="store_true", help="Enable directory mode."
    )
    parser.add_argument(
        "--outputpath", "-o", type=str, required=True, help="Specify the output path."
    )

    args = parser.parse_args()

    # Check if the specified path exists
    if not os.path.exists(args.path):
        raise argparse.ArgumentError(None, f"Path '{args.path}' does not exist.")

    # Check if the specified path is a directory if directory_mode is enabled
    if args.directory_mode and not os.path.isdir(args.path):
        raise argparse.ArgumentError(None, f"Path '{args.path}' is not a directory.")

    # Check if the specified path is a file if directory_mode is not enabled
    if not args.directory_mode and not os.path.isfile(args.path):
        raise argparse.ArgumentError(None, f"Path '{args.path}' is not a file.")

    # Create the output directory if it doesn't exist
    os.makedirs(args.outputpath, exist_ok=True)

    return args


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    # Call the main function with the parsed arguments
    main(
        args.path,
        args.modelpath,
        args.n_pre,
        args.n_post,
        args.directory_mode,
        args.outputpath,
    )
