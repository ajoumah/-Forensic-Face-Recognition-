from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from PIL import Image


VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff"
)


class EmbeddingEngine:
    """
    General embedding engine for CNNs, ViTs,
    face recognition models, and metric learning models.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        transform: Callable,
        device: Optional[str] = None,
        normalize_embeddings: bool = True
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = model.to(self.device)
        self.model.eval()

        self.transform = transform
        self.normalize_embeddings = normalize_embeddings

    # =====================================
    # IMAGE → EMBEDDING
    # =====================================
    def get_embedding(
        self,
        image_path: str
    ) -> Optional[np.ndarray]:

        try:

            img = Image.open(image_path).convert("RGB")

            tensor = (
                self.transform(img)
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():

                embedding = self.model(tensor)

                # Handle tuple outputs (some models return extra data)
                if isinstance(embedding, tuple):
                    embedding = embedding[0]

            embedding = embedding.squeeze().cpu().numpy()

            embedding = embedding.astype(np.float32)

            # Normalize embedding
            if self.normalize_embeddings:

                norm = np.linalg.norm(embedding)

                if norm > 0:
                    embedding = embedding / norm

            return embedding

        except Exception as e:

            print(f"❌ Failed embedding: {image_path}")
            print(e)

            return None

    # =====================================
    # BUILD FAISS INDEX
    # =====================================
    def build_faiss_index(
        self,
        folder_path: str
    ) -> Tuple[faiss.IndexFlatIP, List[str]]:

        embeddings = []
        filenames = []

        for file in os.listdir(folder_path):

            if not file.lower().endswith(VALID_EXTENSIONS):
                continue

            image_path = os.path.join(folder_path, file)

            embedding = self.get_embedding(image_path)

            if embedding is not None:

                embeddings.append(embedding)
                filenames.append(file)

        if len(embeddings) == 0:
            raise ValueError(
                f"No embeddings generated from: {folder_path}"
            )

        embeddings_np = np.array(embeddings).astype(np.float32)

        # Cosine similarity via normalized vectors
        faiss.normalize_L2(embeddings_np)

        dimension = embeddings_np.shape[1]

        index = faiss.IndexFlatIP(dimension)

        index.add(embeddings_np)

        return index, filenames

    # =====================================
    # FOLDER COMPARISON
    # =====================================
    def compare_folders(
        self,
        source_folder: str,
        destination_folder: str,
        top_k: int = 1
    ) -> Dict:

        print("📦 Building FAISS index...")

        index, destination_files = self.build_faiss_index(
            destination_folder
        )

        results = []

        same_name_matches = 0
        different_name_matches = 0

        for file in os.listdir(source_folder):

            if not file.lower().endswith(VALID_EXTENSIONS):
                continue

            source_path = os.path.join(source_folder, file)

            embedding = self.get_embedding(source_path)

            if embedding is None:
                continue

            embedding = embedding.reshape(1, -1)

            faiss.normalize_L2(embedding)

            distances, indices = index.search(
                embedding,
                top_k
            )

            best_match = destination_files[
                indices[0][0]
            ]

            similarity = float(distances[0][0])

            is_same = file == best_match

            if is_same:
                same_name_matches += 1
            else:
                different_name_matches += 1

            result = {
                "source": file,
                "best_match": best_match,
                "similarity": similarity,
                "same_filename": is_same
            }

            results.append(result)

            print("\n-------------------")
            print(f"Source: {file}")
            print(f"Best Match: {best_match}")
            print(f"Similarity: {similarity:.4f}")

        summary = {
            "total_compared": len(results),
            "same_name_matches": same_name_matches,
            "different_name_matches": different_name_matches,
            "results": results
        }

        return summary