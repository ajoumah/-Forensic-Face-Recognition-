import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_detector_results(counters):
    """
    Plot comparison of face detectors.
    """

    data = []

    for detector, stats in counters.items():
        data.append({
            "Detector": detector,
            "Status": "Detected",
            "Count": stats["found"]
        })
        data.append({
            "Detector": detector,
            "Status": "Not Detected",
            "Count": stats["not_found"]
        })

    df = pd.DataFrame(data)

    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        data=df,
        x="Detector",
        y="Count",
        hue="Status"
    )

    ax.set_title("Face Detection Comparison")
    ax.set_ylabel("Number of Images")

    plt.tight_layout()
    plt.show()