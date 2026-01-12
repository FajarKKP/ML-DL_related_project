import cv2
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
import pathlib as Path


def convolution_2d(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not fine image at path: {path}")
    
    

    


def main():
    parser = argparse.ArgumentParser(
        description="2d convolution from scratch"
    )

    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image"
    )

    args = parser.parse_args()

    try:
        convolution_2d(Path(args.image_path))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()