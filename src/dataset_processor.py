import os
import cv2
from insightface.app import FaceAnalysis
from facenet_pytorch import MTCNN

from src.face_cropper import crop_face


def save_retina_faces(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    # Models
    retina = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    retina.prepare(ctx_id=0)

    mtcnn = MTCNN(keep_all=True, device="cpu")

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
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]

        retina_faces = retina.get(img_rgb)
        mtcnn_boxes, _ = mtcnn.detect(img_rgb)

        retina_found = len(retina_faces) > 0
        mtcnn_found = mtcnn_boxes is not None and len(mtcnn_boxes) > 0

        if retina_found and mtcnn_found:

            stats["both"] += 1

            face = retina_faces[0].bbox
            crop = crop_face(img, face, w, h)

            if crop is not None:
                cv2.imwrite(os.path.join(output_dir, filename), crop)

        elif retina_found:

            stats["retina_only"] += 1

            face = retina_faces[0].bbox
            crop = crop_face(img, face, w, h)

            if crop is not None:
                cv2.imwrite(os.path.join(output_dir, filename), crop)

        elif mtcnn_found:

            stats["mtcnn_only"] += 1

        else:
            stats["none"] += 1

    print("\n📊 Summary:")
    for k, v in stats.items():
        print(f"{k}: {v}")

    return stats