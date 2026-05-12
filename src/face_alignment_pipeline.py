import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

from src.face_alignment import align_face_by_eyes_nose


def align_faces_in_folder(
    input_dir: str,
    output_dir: str
) -> None:
    """
    Detect and align faces in a folder using InsightFace + custom alignment.

    Args:
        input_dir (str): Input image folder
        output_dir (str): Output folder for aligned faces
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize model (DO NOT move inside loop)
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0)

    image_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    for image_path in tqdm(image_files):

        img = cv2.imread(str(image_path))

        if img is None:
            print(f"Warning: could not read {image_path.name}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = app.get(img_rgb)

        if not faces:
            print(f"No face detected in {image_path.name}")
            continue

        face = faces[0]

        # Bounding box (optional use)
        x1, y1, x2, y2 = face.bbox.astype(int)
        face_box = (x1, y1, x2 - x1, y2 - y1)

        # Keypoints
        left_eye = tuple(map(int, face.kps[0]))
        right_eye = tuple(map(int, face.kps[1]))
        nose = tuple(map(int, face.kps[2]))

        try:
            aligned_img, angle = align_face_by_eyes_nose(
                img_rgb,
                left_eye,
                right_eye,
                nose
            )

        except Exception as e:
            print(f"Error aligning {image_path.name}: {e}")
            continue

        aligned_bgr = cv2.cvtColor(aligned_img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(
            str(output_path / image_path.name),
            aligned_bgr
        )

    print(f"✅ Done. Saved to: {output_dir}")