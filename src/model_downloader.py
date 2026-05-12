import shutil
import subprocess
from pathlib import Path


def download_huggingface_model(
    repo_url: str,
    target_dir: str,
    overwrite: bool = False
) -> None:
    """
    Download a Hugging Face model repository using git clone.

    Args:
        repo_url:
            Hugging Face git repository URL.

        target_dir:
            Local directory to save model.

        overwrite:
            If True, remove existing folder first.
    """

    target_path = Path(target_dir)

    # Remove existing directory if requested
    if overwrite and target_path.exists():

        print(f"Removing existing directory: {target_path}")

        shutil.rmtree(target_path)

    # Create parent directory
    target_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Skip if already exists
    if target_path.exists():

        print(f"Model already exists: {target_path}")
        return

    print("Downloading model...")

    subprocess.run(
        [
            "git",
            "clone",
            repo_url,
            str(target_path)
        ],
        check=True
    )

    print(f"✅ Model downloaded to: {target_path}")