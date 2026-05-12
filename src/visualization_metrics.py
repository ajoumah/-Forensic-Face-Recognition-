from typing import Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt


def plot_face_detection_methods(
    methods_results: Dict[str, Tuple[float, float, float]]
) -> None:
    """
    Plot comparison of face detection performance metrics.

    Args:
        methods_results:
            Dictionary in format:
            {
                "MethodName": (postmortem, antemortem, overall),
                ...
            }

    Returns:
        None
    """

    if not methods_results:
        raise ValueError("methods_results is empty")

    methods: List[str] = list(methods_results.keys())

    postmortem = [methods_results[m][0] for m in methods]
    antemortem = [methods_results[m][1] for m in methods]
    overall = [methods_results[m][2] for m in methods]

    x = np.arange(len(methods))
    width = 0.25

    plt.figure(figsize=(12, 6))

    bars1 = plt.bar(x - width, postmortem, width, label="Postmortem Avg")
    bars2 = plt.bar(x, antemortem, width, label="Antemortem Avg")
    bars3 = plt.bar(x + width, overall, width, label="Overall Avg")

    # Value labels
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.5,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.ylabel("Detection Accuracy (%)")
    plt.xlabel("Face Detection Method")
    plt.title("Face Detection Performance Comparison")

    plt.xticks(x, methods)
    plt.ylim(0, 110)

    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()