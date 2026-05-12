from pathlib import Path
import shutil
import re
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)


def sort_and_copy_faces(
    main_dir: str,
    output_dead: str = "dead_faces",
    output_alive: str = "alive_faces"
) -> None:
    """
    Sort and copy face images into separate folders for dead and alive faces.

    Args:
        main_dir (str): Root directory containing image files.
        output_dead (str): Output directory for dead face images.
        output_alive (str): Output directory for alive face images.

    Returns:
        None
    """

    main_path = Path(main_dir)
    dead_path = Path(output_dead)
    alive_path = Path(output_alive)

    # Create output directories
    dead_path.mkdir(parents=True, exist_ok=True)
    alive_path.mkdir(parents=True, exist_ok=True)

    # Regex patterns
    pattern_dead = re.compile(
        r"^personD(?:0{0,2})(\d{1,3})\.jpg$",
        re.IGNORECASE
    )

    pattern_alive = re.compile(
        r"^PersonA(?:0{0,2}|a0|aa|ab)(\d{1,3})\.jpg$",
        re.IGNORECASE
    )

    copied_dead = 0
    copied_alive = 0

    # Traverse directories
    for file_path in main_path.rglob("*.jpg"):

        filename = file_path.name

        try:
            # Dead faces
            if pattern_dead.match(filename):
                shutil.copy2(file_path, dead_path / filename)
                copied_dead += 1

            # Alive faces
            elif pattern_alive.match(filename):
                shutil.copy2(file_path, alive_path / filename)
                copied_alive += 1

        except Exception as error:
            logging.error(f"Failed to process {file_path}: {error}")

    logging.info(f"Copied {copied_dead} dead face images")
    logging.info(f"Copied {copied_alive} alive face images")


if __name__ == "__main__":

    DATASET_PATH = "dataset"

    sort_and_copy_faces(
        main_dir=DATASET_PATH,
        output_dead="output/dead_faces",
        output_alive="output/alive_faces"
    )