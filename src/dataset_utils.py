import os
import shutil
from typing import List


def copy_images_in_range(
    input_folder: str,
    output_folder: str,
    start: int,
    end: int,
    base_name: str = "person",
    extensions: List[str] = [".jpg", ".jpeg", ".png"]
) -> int:
    """
    Copy images in a numeric range from one folder to another.

    Args:
        input_folder: Source directory
        output_folder: Destination directory
        start: Start index
        end: End index
        base_name: File prefix (default: person)
        extensions: Allowed image extensions

    Returns:
        Number of copied files
    """

    os.makedirs(output_folder, exist_ok=True)

    copied_count = 0

    for i in range(start, end + 1):

        filename_base = f"{base_name}{i:03d}"

        found = False

        for ext in extensions:

            input_path = os.path.join(input_folder, filename_base + ext)

            if os.path.isfile(input_path):

                output_path = os.path.join(output_folder, filename_base + ext)

                shutil.copy2(input_path, output_path)

                print(f"Copied: {input_path} → {output_path}")

                copied_count += 1
                found = True
                break

        if not found:
            print(f"⚠️ Missing: {filename_base}")

    return copied_count