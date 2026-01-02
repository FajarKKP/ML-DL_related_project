import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

def k_means_segment(image, k):
    h, w, c = image.shape
    pixels = image.reshape((-1, c)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2,
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )

    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    return segmented.reshape((h,w,c))


def main(image_path, k):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"Could not load path: {image_path}")
        sys.exit(1)
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
     # ---- Convert color spaces ----
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    # ---- Segment ----
    seg_rgb = k_means_segment(img_rgb, k)
    seg_hsv = k_means_segment(img_hsv, k)
    seg_lab = k_means_segment(img_lab, k)

    # Convert results back to RGB for display
    seg_hsv_rgb = cv2.cvtColor(seg_hsv, cv2.COLOR_HSV2RGB)
    seg_lab_rgb = cv2.cvtColor(seg_lab, cv2.COLOR_LAB2RGB)

    # ---- Display ----
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original (RGB)")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(seg_rgb)
    plt.title("k-means on RGB")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(seg_hsv_rgb)
    plt.title("k-means on HSV")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(seg_lab_rgb)
    plt.title("k-means on Lab")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare k-means segmentation in RGB, HSV, and Lab"
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of clusters",
    )

    args = parser.parse_args()
    main(args.image, args.k)