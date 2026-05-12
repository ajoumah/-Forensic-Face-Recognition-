import os
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


class RetrievalEngine:
    """
    General retrieval and comparison engine
    for biometric embeddings, CNNs, ViTs,
    metric learning, and face recognition.
    """

    def __init__(
        self,
        embedding_function: Callable,
        id_extractor: Optional[Callable] = None,
        normalize: bool = True
    ):

        self.embedding_function = embedding_function
        self.id_extractor = id_extractor
        self.normalize = normalize

    # =====================================
    # INDEX DATASET
    # =====================================
    def build_index(
        self,
        folder_path: str
    ):

        embeddings = []
        filenames = []
        identities = []

        folder = Path(folder_path)

        for file in folder.iterdir():

            if not file.is_file():
                continue

            if file.suffix.lower() not in VALID_EXTENSIONS:
                continue

            identity = None

            if self.id_extractor:
                identity = self.id_extractor(file.name)

                if identity is None:
                    continue

            embedding = self.embedding_function(str(file))

            if embedding is None:
                continue

            embedding = embedding.astype(np.float32)

            embeddings.append(embedding)
            filenames.append(file.name)
            identities.append(identity)

        if len(embeddings) == 0:
            raise ValueError(
                f"No embeddings found in: {folder_path}"
            )

        embeddings_np = np.array(embeddings)

        if self.normalize:
            faiss.normalize_L2(embeddings_np)

        dimension = embeddings_np.shape[1]

        index = faiss.IndexFlatIP(dimension)

        index.add(embeddings_np)

        return {
            "index": index,
            "filenames": filenames,
            "identities": identities
        }

    # =====================================
    # COMPARE DATASETS
    # =====================================
    def compare_folders(
        self,
        source_folder: str,
        destination_folder: str,
        top_k: int = 1
    ) -> Dict:

        print("📦 Building destination index...")

        destination_data = self.build_index(
            destination_folder
        )

        index = destination_data["index"]

        destination_filenames = (
            destination_data["filenames"]
        )

        destination_ids = (
            destination_data["identities"]
        )

        results = []

        correct_matches = 0
        incorrect_matches = 0

        source_folder = Path(source_folder)

        for file in source_folder.iterdir():

            if not file.is_file():
                continue

            if file.suffix.lower() not in VALID_EXTENSIONS:
                continue

            source_id = None

            if self.id_extractor:
                source_id = self.id_extractor(file.name)

                if source_id is None:
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
                faiss.normalize_L2(embedding)

            distances, indices = index.search(
                embedding,
                top_k
            )

            best_index = indices[0][0]

            best_match = (
                destination_filenames[best_index]
            )

            best_identity = (
                destination_ids[best_index]
            )

            similarity = float(distances[0][0])

            is_correct = (
                source_id == best_identity
            )

            if is_correct:
                correct_matches += 1
            else:
                incorrect_matches += 1

            result = {
                "source_file": file.name,
                "source_id": source_id,
                "best_match": best_match,
                "best_match_id": best_identity,
                "similarity": similarity,
                "correct_match": is_correct
            }

            results.append(result)

            print("\n---------------------")
            print(f"Source: {file.name}")
            print(f"Source ID: {source_id}")
            print(f"Best Match: {best_match}")
            print(f"Similarity: {similarity:.4f}")
            print(f"Correct: {is_correct}")

        summary = {
            "total_compared": len(results),
            "correct_matches": correct_matches,
            "incorrect_matches": incorrect_matches,
            "accuracy": (
                correct_matches / max(len(results), 1)
            ),
            "results": results
        }

        return summary