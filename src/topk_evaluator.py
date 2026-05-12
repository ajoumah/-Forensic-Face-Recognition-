from pathlib import Path
from typing import Callable, Dict, List, Optional

import faiss
import numpy as np


VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff"
)


class TopKEvaluator:
    """
    General Top-K retrieval evaluator
    for CNNs, ViTs, metric learning,
    face recognition, and embedding systems.
    """

    def __init__(
        self,
        embedding_function: Callable,
        id_extractor: Callable,
        normalize: bool = True,
        metric: str = "cosine"
    ):

        self.embedding_function = embedding_function
        self.id_extractor = id_extractor
        self.normalize = normalize
        self.metric = metric

    # =====================================
    # CREATE FAISS INDEX
    # =====================================
    def build_index(
        self,
        folder_path: str
    ) -> Dict:

        embeddings = []
        filenames = []
        identities = []

        folder = Path(folder_path)

        for file in folder.iterdir():

            if not file.is_file():
                continue

            if file.suffix.lower() not in VALID_EXTENSIONS:
                continue

            identity = self.id_extractor(
                file.name
            )

            if identity is None:
                continue

            embedding = self.embedding_function(
                str(file)
            )

            if embedding is None:
                continue

            embedding = embedding.astype(
                np.float32
            )

            embeddings.append(embedding)
            filenames.append(file.name)
            identities.append(identity)

        if len(embeddings) == 0:
            raise ValueError(
                f"No embeddings found in: {folder_path}"
            )

        embeddings_np = np.array(
            embeddings,
            dtype=np.float32
        )

        if self.normalize:
            faiss.normalize_L2(
                embeddings_np
            )

        dimension = embeddings_np.shape[1]

        # Cosine similarity
        if self.metric == "cosine":

            index = faiss.IndexFlatIP(
                dimension
            )

        # Euclidean distance
        elif self.metric == "l2":

            index = faiss.IndexFlatL2(
                dimension
            )

        else:
            raise ValueError(
                f"Unsupported metric: {self.metric}"
            )

        index.add(embeddings_np)

        return {
            "index": index,
            "filenames": filenames,
            "identities": identities
        }

    # =====================================
    # EVALUATE TOP-K
    # =====================================
    def evaluate(
        self,
        query_folder: str,
        gallery_folder: str,
        top_k_values: List[int] = [1, 5]
    ) -> Dict:

        print("📦 Building gallery index...")

        gallery_data = self.build_index(
            gallery_folder
        )

        index = gallery_data["index"]

        gallery_ids = (
            gallery_data["identities"]
        )

        max_k = max(top_k_values)

        topk_correct = {
            k: 0 for k in top_k_values
        }

        total_queries = 0

        detailed_results = []

        query_folder = Path(query_folder)

        for file in query_folder.iterdir():

            if not file.is_file():
                continue

            if file.suffix.lower() not in VALID_EXTENSIONS:
                continue

            query_id = self.id_extractor(
                file.name
            )

            if query_id is None:
                continue

            embedding = self.embedding_function(
                str(file)
            )

            if embedding is None:
                continue

            embedding = (
                embedding
                .astype(np.float32)
                .reshape(1, -1)
            )

            if self.normalize:
                faiss.normalize_L2(
                    embedding
                )

            distances, indices = index.search(
                embedding,
                max_k
            )

            retrieved_ids = [
                gallery_ids[idx]
                for idx in indices[0]
            ]

            result = {
                "query_file": file.name,
                "query_id": query_id,
                "retrieved_ids": retrieved_ids
            }

            detailed_results.append(result)

            # Evaluate each K
            for k in top_k_values:

                topk_ids = retrieved_ids[:k]

                if query_id in topk_ids:
                    topk_correct[k] += 1

            total_queries += 1

            print("\n----------------")
            print(f"Query: {file.name}")
            print(f"Query ID: {query_id}")
            print(f"Retrieved IDs: {retrieved_ids}")

        # Final metrics
        metrics = {}

        for k in top_k_values:

            accuracy = (
                topk_correct[k]
                / max(total_queries, 1)
            )

            metrics[f"top_{k}_accuracy"] = (
                accuracy
            )

        summary = {
            "total_queries": total_queries,
            "metrics": metrics,
            "results": detailed_results
        }

        return summary