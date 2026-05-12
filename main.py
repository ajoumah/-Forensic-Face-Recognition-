"""
========================================================
Face Recognition Research Framework - Main Entry Point
========================================================
"""

# =========================
# GLOBAL IMPORTS
# =========================
import cv2
import numpy as np
from torchvision import transforms

# =========================
# SRC IMPORTS
# =========================
from src.image_rotator import rotate_images
from src.file_utils import count_images_in_folder
from src.face_alignment import align_face_by_eyes_nose

from src.detectors import load_detectors
from src.face_comparison import compare_face_detectors
from src.visualization import plot_detector_results
from src.visualization_metrics import plot_face_detection_methods
from src.dataset_analysis import analyze_image_dimensions

from src.model_downloader import download_huggingface_model
from src.model_wrapper_patch import load_model_config
from src.model_loader import load_model

from src.embedding_engine import EmbeddingEngine
from src.retrieval_engine import RetrievalEngine
from src.verification_evaluator import VerificationEvaluator
from src.topk_evaluator import TopKEvaluator

from src.embedding_analysis import EmbeddingAnalyzer

from src.backbone_factory import GenericEmbeddingModel

from src.id_extractors import numeric_filename_extractor


# ========================================================
# CONFIG SECTION (CHANGE ONLY THIS FOR NEW EXPERIMENTS)
# ========================================================
CONFIG = {

    # Paths
    "data_dir": "data",
    "source_dir": "data/source",
    "gallery_dir": "data/gallery",
    "query_dir": "data/query",
    "destination_dir": "data/destination",

    # Model config
    "backbone": "resnet50",
    "image_size": 224,

    # Rotation
    "rotate_output_90": "output/rotated_90",
    "rotate_output_180": "output/rotated_180",

    # Flags (enable/disable experiments)
    "run_detection": True,
    "run_retrieval": True,
    "run_verification": True,
    "run_topk": True,
    "run_analysis": True,
}


# ========================================================
# COMMON TRANSFORM
# ========================================================
def get_transform(size: int):

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5] * 3,
            std=[0.5] * 3
        )
    ])


# ========================================================
# MODEL FACTORY
# ========================================================
def build_model():

    return GenericEmbeddingModel(
        backbone_name=CONFIG["backbone"]
    )


# ========================================================
# PIPELINE 1: DATA PREPROCESSING
# ========================================================
def run_preprocessing():

    print("\n Running preprocessing...")

    rotate_images(
        input_folder="output/dead_faces",
        output_folder=CONFIG["rotate_output_90"],
        angle=-90
    )

    rotate_images(
        input_folder="output/dead_faces",
        output_folder=CONFIG["rotate_output_180"],
        angle=180
    )

    print(
        count_images_in_folder("output/dead_faces")
    )

    print(
        count_images_in_folder("output/alive_faces")
    )


# ========================================================
# PIPELINE 2: FACE DETECTION BENCHMARK
# ========================================================
def run_detection():

    print("\n Running face detection benchmark...")

    detectors = load_detectors()

    results = compare_face_detectors(
        CONFIG["data_dir"],
        detectors
    )

    plot_detector_results(results)


# ========================================================
# PIPELINE 3: DATASET ANALYSIS
# ========================================================
def run_dataset_analysis():

    print("\n Dataset analysis...")

    summary = analyze_image_dimensions(
        CONFIG["data_dir"]
    )

    print(summary)


# ========================================================
# PIPELINE 4: RETRIEVAL (FAISS / TOP-K)
# ========================================================
def run_retrieval():

    print("\n Running retrieval pipeline...")

    model = build_model()
    transform = get_transform(CONFIG["image_size"])

    engine = EmbeddingEngine(
        model=model,
        transform=transform
    )

    retrieval = RetrievalEngine(
        embedding_function=engine.get_embedding,
        id_extractor=numeric_filename_extractor
    )

    results = retrieval.compare_folders(
        source_folder=CONFIG["query_dir"],
        destination_folder=CONFIG["gallery_dir"]
    )

    print(results["accuracy"])


# ========================================================
# PIPELINE 5: TOP-K EVALUATION
# ========================================================
def run_topk():

    print("\n Running Top-K evaluation...")

    model = build_model()
    transform = get_transform(CONFIG["image_size"])

    engine = EmbeddingEngine(
        model=model,
        transform=transform
    )

    evaluator = TopKEvaluator(
        embedding_function=engine.get_embedding,
        id_extractor=numeric_filename_extractor
    )

    results = evaluator.evaluate(
        query_folder=CONFIG["query_dir"],
        gallery_folder=CONFIG["gallery_dir"],
        top_k_values=[1, 5, 10]
    )

    print(results["metrics"])


# ========================================================
# PIPELINE 6: VERIFICATION (ROC / EER)
# ========================================================
def run_verification():

    print("\n Running verification...")

    model = build_model()
    transform = get_transform(CONFIG["image_size"])

    engine = EmbeddingEngine(
        model=model,
        transform=transform
    )

    evaluator = VerificationEvaluator(
        embedding_function=engine.get_embedding,
        id_extractor=numeric_filename_extractor
    )

    results = evaluator.evaluate(
        query_folder=CONFIG["query_dir"],
        gallery_folder=CONFIG["gallery_dir"],
        save_path="outputs/results"
    )

    print("AUC:", results["auc"])
    print("EER:", results["eer"])

    evaluator.plot_roc(
        results["fpr"],
        results["tpr"],
        results["auc"]
    )


# ========================================================
# PIPELINE 7: EMBEDDING ANALYSIS
# ========================================================
def run_embedding_analysis():

    print("\n Running embedding analysis...")

    analyzer = EmbeddingAnalyzer(
        normalize=True
    )

    results = analyzer.analyze(
        source_embeddings=alive_embeddings,
        target_embeddings=dead_embeddings
    )

    analyzer.print_summary(results)


# ========================================================
# MAIN RUNNER
# ========================================================
def main():

    print("\n Starting Face Recognition Framework")

    run_preprocessing()

    if CONFIG["run_detection"]:
        run_detection()

    run_dataset_analysis()

    if CONFIG["run_retrieval"]:
        run_retrieval()

    if CONFIG["run_topk"]:
        run_topk()

    if CONFIG["run_verification"]:
        run_verification()

    if CONFIG["run_analysis"]:
        run_embedding_analysis()

    print("\n✅ All experiments completed successfully")


# ========================================================
# ENTRY POINT
# ========================================================
if __name__ == "__main__":
    main()