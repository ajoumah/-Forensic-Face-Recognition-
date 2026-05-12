from typing import Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_detection_rate_bar_chart(
    counters: Dict[str, Dict[str, int]]
) -> None:
    """
    Plot detection rate (%) for multiple face detectors.

    Args:
        counters:
            Dictionary format:
            {
                "DetectorName": {"found": int, "not_found": int},
                ...
            }

    Returns:
        None
    """

    if not counters:
        raise ValueError("counters is empty")

    detectors = list(counters.keys())

    detection_rates = []

    for d in detectors:
        total = counters[d]["found"] + counters[d]["not_found"]

        if total == 0:
            rate = 0
        else:
            rate = (counters[d]["found"] / total) * 100

        detection_rates.append(rate)

    # Fixed professional color palette
    colors = [
        "#4CAF50", "#2196F3", "#FF5722", "#9C27B0",
        "#FFC107", "#00BCD4"
    ]

    # Expand colors if needed (deterministic fallback)
    if len(detectors) > len(colors):
        colors.extend(
            ["#777777"] * (len(detectors) - len(colors))
        )

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        detectors,
        detection_rates,
        color=colors[:len(detectors)]
    )

    # Value labels
    for bar, rate in zip(bars, detection_rates):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.ylabel("Detection Rate (%)")
    plt.ylim(0, 100)
    plt.title("Face Detection Rate Comparison")

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Clean legend
    legend_handles = [
        Patch(color=colors[i], label=detectors[i])
        for i in range(len(detectors))
    ]

    plt.legend(
        handles=legend_handles,
        title="Face Detectors",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()