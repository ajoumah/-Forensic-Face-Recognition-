import os
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image


VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff"
)


def analyze_image_dimensions(
    folder_path: str
) -> pd.DataFrame:
    """
    Analyze image dimensions in a dataset folder.

    Args:
        folder_path:
            Path to image dataset folder.

    Returns:
        pandas.DataFrame:
            Statistical summary of image widths and heights.
    """

    widths: List[int] = []
    heights: List[int] = []

    for root, _, files in os.walk(folder_path):

        for file in files:

            if file.lower().endswith(VALID_EXTENSIONS):

                image_path = os.path.join(root, file)

                try:
                    with Image.open(image_path) as img:

                        width, height = img.size

                        widths.append(width)
                        heights.append(height)

                except Exception as e:
                    print(f"⚠️ Error reading {file}: {e}")

    # Validation
    if len(widths) == 0:
        raise ValueError(
            f"No valid images found in: {folder_path}"
        )

    widths_np = np.array(widths)
    heights_np = np.array(heights)

    summary = pd.DataFrame({
        "Dimension": ["Width", "Height"],
        "Mean": [
            np.mean(widths_np),
            np.mean(heights_np)
        ],
        "Median": [
            np.median(widths_np),
            np.median(heights_np)
        ],
        "Min": [
            np.min(widths_np),
            np.min(heights_np)
        ],
        "Max": [
            np.max(widths_np),
            np.max(heights_np)
        ],
        "Std": [
            np.std(widths_np),
            np.std(heights_np)
        ]
    })

    return summary