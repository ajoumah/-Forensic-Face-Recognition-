from pathlib import Path
import yaml


def load_model_config(model_root: str):
    """
    Load model.yaml safely using dynamic paths.

    Args:
        model_root:
            Root directory of model repository.

    Returns:
        dict: YAML configuration
    """

    model_path = Path(model_root)

    yaml_path = (
        model_path
        / "pretrained_model"
        / "model.yaml"
    )

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Missing config file: {yaml_path}"
        )

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    return dict(config)