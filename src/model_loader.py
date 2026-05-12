from pathlib import Path
from transformers import AutoModel


def load_model(model_dir: str):
    """
    Load AdaFace model from local directory.

    Args:
        model_dir:
            Path to model repository.

    Returns:
        Loaded Hugging Face model.
    """

    model_path = Path(model_dir)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}"
        )

    print(f"Loading model from: {model_path}")

    model = AutoModel.from_pretrained(
        str(model_path),
        trust_remote_code=True
    )

    print("✅ Model loaded successfully")

    return model