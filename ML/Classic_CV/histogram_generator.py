import cv2 
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def show_img_histogram(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not load image from path: {image_path}")

    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

    #Plot image
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.plot(histogram)
    plt.title("Intensity Histogram")
    plt.xlabel("Intensity Value")
    plt.ylabel("Pixel Count")
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Display the histogram of an image"
    )
    
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image"
    )

    args = parser.parse_args()

    try:
        show_img_histogram(args.image_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()