from pathlib import Path
from PIL import Image
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def rotate_images(
    input_folder: str,
    output_folder: str,
    angle: int
) -> None:
    """
    Rotate all images in a folder by a specified angle.

    Args:
        input_folder (str): Folder containing input images.
        output_folder (str): Folder to save rotated images.
        angle (int): Rotation angle in degrees.

    Returns:
        None
    """

    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    rotated_count = 0

    # Recursive search
    for image_path in input_path.rglob("*"):

        # Skip unsupported files
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:

            output_file = output_path / image_path.name

            with Image.open(image_path) as img:

                rotated = img.rotate(angle, expand=True)

                rotated.save(output_file)

            rotated_count += 1

            logging.info(f"Rotated: {image_path.name}")

        except Exception as error:

            logging.error(
                f"Failed to process {image_path.name}: {error}"
            )

    logging.info(f"Total rotated images: {rotated_count}")