import numpy as np

from typing import Dict, Any
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingAnalyzer:

    """
    Analyze relationships between two embedding sets.

    Supports:
    - Any embedding dimension
    - Any backbone
    - Face embeddings
    - Biometric embeddings
    - CNN / ViT embeddings
    """

    def __init__(
        self,
        normalize: bool = True
    ):

        self.normalize = normalize

    # ==========================================
    # NORMALIZATION
    # ==========================================
    def _normalize_embeddings(
        self,
        embeddings: np.ndarray
    ) -> np.ndarray:

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        norms = np.clip(
            norms,
            1e-12,
            None
        )

        return embeddings / norms

    # ==========================================
    # MAIN ANALYSIS
    # ==========================================
    def analyze(
        self,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray
    ) -> Dict[str, Any]:

        """
        Analyze transformation between two
        embedding spaces.

        Parameters
        ----------
        source_embeddings : np.ndarray
            Shape: [N, D]

        target_embeddings : np.ndarray
            Shape: [N, D]

        Returns
        -------
        Dict[str, Any]
        """

        # ======================================
        # VALIDATION
        # ======================================
        if source_embeddings.shape != target_embeddings.shape:

            raise ValueError(
                "Shape mismatch between "
                "source and target embeddings."
            )

        if len(source_embeddings.shape) != 2:

            raise ValueError(
                "Embeddings must have shape [N, D]."
            )

        x = source_embeddings.astype(np.float32)
        y = target_embeddings.astype(np.float32)

        # ======================================
        # OPTIONAL NORMALIZATION
        # ======================================
        if self.normalize:

            x = self._normalize_embeddings(x)
            y = self._normalize_embeddings(y)

        n_samples, embedding_dim = x.shape

        # ======================================
        # COSINE SIMILARITY
        # ======================================
        cosine_scores = np.array([
            cosine_similarity(
                x[i].reshape(1, -1),
                y[i].reshape(1, -1)
            )[0][0]
            for i in range(n_samples)
        ])

        mean_cosine_similarity = np.mean(
            cosine_scores
        )

        std_cosine_similarity = np.std(
            cosine_scores
        )

        # ======================================
        # EUCLIDEAN DISTANCE
        # ======================================
        euclidean_distances = np.linalg.norm(
            x - y,
            axis=1
        )

        mean_euclidean_distance = np.mean(
            euclidean_distances
        )

        std_euclidean_distance = np.std(
            euclidean_distances
        )

        # ======================================
        # TRANSLATION VECTORS
        # ======================================
        translation_vectors = y - x

        mean_translation_vector = np.mean(
            translation_vectors,
            axis=0
        )

        translation_vector_norm = np.linalg.norm(
            mean_translation_vector
        )

        shift_magnitudes = np.linalg.norm(
            translation_vectors,
            axis=1
        )

        mean_shift_magnitude = np.mean(
            shift_magnitudes
        )

        std_shift_magnitude = np.std(
            shift_magnitudes
        )

        translation_variance = np.var(
            translation_vectors,
            axis=0
        )

        translation_std = np.std(
            translation_vectors,
            axis=0
        )

        # ======================================
        # PEARSON CORRELATION
        # ======================================
        pearson_coefficients = []

        for dim in range(embedding_dim):

            try:

                corr, _ = pearsonr(
                    x[:, dim],
                    y[:, dim]
                )

                if np.isnan(corr):

                    corr = 0.0

            except Exception:

                corr = 0.0

            pearson_coefficients.append(corr)

        pearson_coefficients = np.array(
            pearson_coefficients
        )

        mean_pearson_correlation = np.mean(
            pearson_coefficients
        )

        std_pearson_correlation = np.std(
            pearson_coefficients
        )

        # ======================================
        # FINAL RESULTS
        # ======================================
        results = {

            # -----------------------------
            # Cosine Similarity
            # -----------------------------
            "cosine_similarities":
                cosine_scores,

            "mean_cosine_similarity":
                mean_cosine_similarity,

            "std_cosine_similarity":
                std_cosine_similarity,

            # -----------------------------
            # Euclidean Distance
            # -----------------------------
            "euclidean_distances":
                euclidean_distances,

            "mean_euclidean_distance":
                mean_euclidean_distance,

            "std_euclidean_distance":
                std_euclidean_distance,

            # -----------------------------
            # Translation Analysis
            # -----------------------------
            "translation_vectors":
                translation_vectors,

            "mean_translation_vector":
                mean_translation_vector,

            "translation_vector_norm":
                translation_vector_norm,

            "shift_magnitudes":
                shift_magnitudes,

            "mean_shift_magnitude":
                mean_shift_magnitude,

            "std_shift_magnitude":
                std_shift_magnitude,

            "translation_variance":
                translation_variance,

            "translation_std":
                translation_std,

            # -----------------------------
            # Pearson Correlation
            # -----------------------------
            "pearson_coefficients":
                pearson_coefficients,

            "mean_pearson_correlation":
                mean_pearson_correlation,

            "std_pearson_correlation":
                std_pearson_correlation,

            # -----------------------------
            # Metadata
            # -----------------------------
            "num_samples":
                n_samples,

            "embedding_dimension":
                embedding_dim
        }

        return results

    # ==========================================
    # PRINT SUMMARY
    # ==========================================
    @staticmethod
    def print_summary(results: Dict[str, Any]):

        print("\n🧪 Embedding Analysis Results")
        print("=" * 50)

        print(
            f"Samples: "
            f"{results['num_samples']}"
        )

        print(
            f"Embedding Dimension: "
            f"{results['embedding_dimension']}"
        )

        print("\n📌 Similarity Metrics")

        print(
            f"Mean Cosine Similarity: "
            f"{results['mean_cosine_similarity']:.4f}"
        )

        print(
            f"Std Cosine Similarity: "
            f"{results['std_cosine_similarity']:.4f}"
        )

        print(
            f"Mean Euclidean Distance: "
            f"{results['mean_euclidean_distance']:.4f}"
        )

        print(
            f"Std Euclidean Distance: "
            f"{results['std_euclidean_distance']:.4f}"
        )

        print("\n📌 Translation Analysis")

        print(
            f"Mean Shift Magnitude: "
            f"{results['mean_shift_magnitude']:.4f}"
        )

        print(
            f"Std Shift Magnitude: "
            f"{results['std_shift_magnitude']:.4f}"
        )

        print(
            f"Translation Vector Norm: "
            f"{results['translation_vector_norm']:.4f}"
        )

        print("\n📌 Correlation Analysis")

        print(
            f"Mean Pearson Correlation: "
            f"{results['mean_pearson_correlation']:.4f}"
        )

        print(
            f"Std Pearson Correlation: "
            f"{results['std_pearson_correlation']:.4f}"
        )