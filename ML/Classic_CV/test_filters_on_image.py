import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path


def apply_filter(path: Path):
    # Load image
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not find image at path: {path}")

    # Convert BGR (OpenCV) -> RGB (matplotlib)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))

    # --------------------
    # Original Image
    # --------------------
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    # --------------------
    # Low-pass (Mean Filter)
    # --------------------
    kernel_size = 5
    low_pass = cv2.blur(image, (kernel_size, kernel_size))

    plt.subplot(2, 2, 2)
    plt.imshow(low_pass)
    plt.title("Low-Pass (Mean Filter)")
    plt.axis("off")

    # --------------------
    # Gaussian Low-pass
    # --------------------
    gaussian_low = cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)

    plt.subplot(2, 2, 3)
    plt.imshow(gaussian_low)
    plt.title("Gaussian Low-Pass")
    plt.axis("off")

    # --------------------
    # High-pass = Image - Gaussian Low-pass
    # --------------------
    high_pass = image.astype(np.float32) - gaussian_low.astype(np.float32)

    # Normalize for display
    high_pass = cv2.normalize(
        high_pass, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    plt.subplot(2, 2, 4)
    plt.imshow(high_pass)
    plt.title("High-Pass (Image - Gaussian)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Apply spatial filters (low-pass, Gaussian, high-pass) to an image"
    )

    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image"
    )

    args = parser.parse_args()

    try:
        apply_filter(Path(args.image_path))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
