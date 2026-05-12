from pathlib import Path
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


def count_images_in_folder(folder_path: str) -> int:
    """
    Count image files recursively inside a folder.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        int: Total number of image files.
    """

    folder = Path(folder_path)

    image_count = sum(
        1
        for file_path in folder.rglob("*")
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    logging.info(
        f"Total images in '{folder_path}': {image_count}"
    )

    return image_count