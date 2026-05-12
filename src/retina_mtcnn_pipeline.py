import os
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from insightface.app import FaceAnalysis


def _safe_crop(img, bbox, img_w, img_h):
    """
    Safely crop image using bounding box.
    """
    x1, y1, x2, y2 = bbox.astype(int)

    x1, y1, x2, y2 = np.clip(
        [x1, y1, x2, y2],
        [0, 0, 0, 0],
        [img_w, img_h, img_w, img_h]
    )

    if x1 >= x2 or y1 >= y2:
        return None

    return img[y1:y2, x1:x2]


def save_retina_and_mtcnn_faces(input_dir, output_dir):
    """
    Save cropped faces using BOTH RetinaFace and MTCNN.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Models
    retina = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    retina.prepare(ctx_id=0)

    mtcnn = MTCNN(keep_all=True, device="cpu")

    # Stats
    stats = {
        "both": 0,
        "retina_only": 0,
        "mtcnn_only": 0,
        "none": 0
    }

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    for filename in image_files:

        path = os.path.join(input_dir, filename)
        img = cv2.imread(path)

        if img is None:
            print(f"⚠️ Skipping unreadable: {filename}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        retina_faces = retina.get(img_rgb)
        mtcnn_boxes, _ = mtcnn.detect(img_rgb)

        retina_found = len(retina_faces) > 0
        mtcnn_found = mtcnn_boxes is not None and len(mtcnn_boxes) > 0

        # -------------------------
        # BOTH DETECTED
        # -------------------------
        if retina_found and mtcnn_found:

            stats["both"] += 1

            # Retina crop
            r_crop = _safe_crop(img, retina_faces[0].bbox, w, h)

            if r_crop is not None:
                cv2.imwrite(
                    os.path.join(output_dir, f"retina_{filename}"),
                    r_crop
                )

            # MTCNN crop
            m_crop = _safe_crop(img, mtcnn_boxes[0], w, h)

            if m_crop is not None:
                cv2.imwrite(
                    os.path.join(output_dir, f"mtcnn_{filename}"),
                    m_crop
                )

        # -------------------------
        # ONLY RETINA
        # -------------------------
        elif retina_found:

            stats["retina_only"] += 1

            r_crop = _safe_crop(img, retina_faces[0].bbox, w, h)

            if r_crop is not None:
                cv2.imwrite(
                    os.path.join(output_dir, f"retina_{filename}"),
                    r_crop
                )

        # -------------------------
        # ONLY MTCNN
        # -------------------------
        elif mtcnn_found:

            stats["mtcnn_only"] += 1

            m_crop = _safe_crop(img, mtcnn_boxes[0], w, h)

            if m_crop is not None:
                cv2.imwrite(
                    os.path.join(output_dir, f"mtcnn_{filename}"),
                    m_crop
                )

        # -------------------------
        # NONE
        # -------------------------
        else:
            stats["none"] += 1
            print(f"❌ No faces found: {filename}")

    # Summary
    print("\n📊 FINAL SUMMARY")
    for k, v in stats.items():
        print(f"{k}: {v}")

    return stats