from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.metrics import (
    roc_curve,
    auc
)


VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff"
)


class VerificationEvaluator:
    """
    General biometric verification evaluator.

    Supports:
    - Face recognition
    - Fingerprint verification
    - Embedding systems
    - CNNs
    - ViTs
    - Metric learning models
    """

    def __init__(
        self,
        embedding_function: Callable,
        id_extractor: Callable,
        similarity_function: Optional[Callable] = None,
        normalize_embeddings: bool = True
    ):

        self.embedding_function = embedding_function

        self.id_extractor = id_extractor

        self.normalize_embeddings = (
            normalize_embeddings
        )

        # Default cosine similarity
        self.similarity_function = (
            similarity_function
            or self.cosine_similarity
        )

    # =====================================
    # DEFAULT COSINE SIMILARITY
    # =====================================
    @staticmethod
    def cosine_similarity(
        emb1: np.ndarray,
        emb2: np.ndarray
    ) -> float:

        return float(
            np.dot(emb1, emb2)
        )

    # =====================================
    # NORMALIZE EMBEDDING
    # =====================================
    @staticmethod
    def normalize_embedding(
        embedding: np.ndarray
    ) -> np.ndarray:

        norm = np.linalg.norm(embedding)

        if norm > 0:
            return embedding / norm

        return embedding

    # =====================================
    # LOAD GALLERY EMBEDDINGS
    # =====================================
    def load_gallery_embeddings(
        self,
        gallery_folder: str
    ) -> Dict:

        gallery_data = {}

        folder = Path(gallery_folder)

        for file in folder.iterdir():

            if not file.is_file():
                continue

            if (
                file.suffix.lower()
                not in VALID_EXTENSIONS
            ):
                continue

            identity = self.id_extractor(
                file.name
            )

            if identity is None:
                continue

            embedding = (
                self.embedding_function(
                    str(file)
                )
            )

            if embedding is None:
                continue

            embedding = embedding.astype(
                np.float32
            )

            if self.normalize_embeddings:
                embedding = (
                    self.normalize_embedding(
                        embedding
                    )
                )

            gallery_data.setdefault(
                identity,
                []
            ).append(embedding)

        return gallery_data

    # =====================================
    # EVALUATE VERIFICATION
    # =====================================
    def evaluate(
        self,
        query_folder: str,
        gallery_folder: str,
        save_path: Optional[str] = None,
        threshold: float = 0.5,
        far_targets: List[float] = [
            1e-2,
            1e-3,
            1e-4
        ]
    ) -> Dict:

        print("📦 Loading gallery embeddings...")

        gallery_data = (
            self.load_gallery_embeddings(
                gallery_folder
            )
        )

        genuine_scores = []
        imposter_scores = []

        y_true = []
        y_score = []

        query_folder = Path(query_folder)

        # =====================================
        # QUERY VS GALLERY
        # =====================================
        for file in query_folder.iterdir():

            if not file.is_file():
                continue

            if (
                file.suffix.lower()
                not in VALID_EXTENSIONS
            ):
                continue

            query_id = self.id_extractor(
                file.name
            )

            if query_id is None:
                continue

            query_embedding = (
                self.embedding_function(
                    str(file)
                )
            )

            if query_embedding is None:
                continue

            query_embedding = (
                query_embedding.astype(
                    np.float32
                )
            )

            if self.normalize_embeddings:
                query_embedding = (
                    self.normalize_embedding(
                        query_embedding
                    )
                )

            # Compare against all gallery IDs
            for (
                gallery_id,
                gallery_embeddings
            ) in gallery_data.items():

                for gallery_embedding in (
                    gallery_embeddings
                ):

                    score = (
                        self.similarity_function(
                            query_embedding,
                            gallery_embedding
                        )
                    )

                    is_genuine = int(
                        query_id == gallery_id
                    )

                    y_true.append(
                        is_genuine
                    )

                    y_score.append(score)

                    if is_genuine:
                        genuine_scores.append(
                            score
                        )
                    else:
                        imposter_scores.append(
                            score
                        )

        # =====================================
        # ROC CURVE
        # =====================================
        fpr, tpr, thresholds = roc_curve(
            y_true,
            y_score
        )

        roc_auc = auc(fpr, tpr)

        # =====================================
        # EER
        # =====================================
        fnr = 1 - tpr

        eer_index = np.nanargmin(
            np.abs(fnr - fpr)
        )

        eer = (
            fpr[eer_index]
            + fnr[eer_index]
        ) / 2

        # =====================================
        # TAR @ FAR
        # =====================================
        tar_results = {}

        for far_target in far_targets:

            idx = np.where(
                fpr <= far_target
            )[0]

            if len(idx) == 0:
                tar = 0.0
            else:
                tar = tpr[idx[-1]]

            tar_results[
                f"TAR@FAR={far_target}"
            ] = tar

        # =====================================
        # THRESHOLD ACCURACY
        # =====================================
        predictions = [
            1 if s >= threshold else 0
            for s in y_score
        ]

        accuracy = np.mean(
            np.array(predictions)
            == np.array(y_true)
        )

        # =====================================
        # SAVE RESULTS
        # =====================================
        if save_path:

            save_path = Path(save_path)

            save_path.mkdir(
                parents=True,
                exist_ok=True
            )

            save_file = (
                save_path
                / "verification_results.pkl"
            )

            with open(save_file, "wb") as f:

                pickle.dump(
                    {
                        "genuine_scores":
                            genuine_scores,

                        "imposter_scores":
                            imposter_scores,

                        "y_true":
                            y_true,

                        "y_score":
                            y_score,

                        "fpr":
                            fpr,

                        "tpr":
                            tpr,

                        "auc":
                            roc_auc,

                        "eer":
                            eer
                    },
                    f
                )

            print(
                f"💾 Results saved to: {save_file}"
            )

        # =====================================
        # SUMMARY
        # =====================================
        results = {
            "auc": roc_auc,
            "eer": eer,
            "accuracy": accuracy,
            "tar_results": tar_results,
            "fpr": fpr,
            "tpr": tpr,
            "genuine_scores": genuine_scores,
            "imposter_scores": imposter_scores
        }

        return results

    # =====================================
    # PLOT ROC
    # =====================================
    @staticmethod
    def plot_roc(
        fpr,
        tpr,
        auc_score,
        title="ROC Curve"
    ):

        plt.figure(figsize=(6, 6))

        plt.plot(
            fpr,
            tpr,
            label=f"AUC = {auc_score:.4f}"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        plt.xlabel(
            "False Positive Rate"
        )

        plt.ylabel(
            "True Positive Rate"
        )

        plt.title(title)

        plt.legend()

        plt.grid(True)

        plt.show()