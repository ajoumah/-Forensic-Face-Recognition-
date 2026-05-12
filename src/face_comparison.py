import os
import cv2


def compare_face_detectors(input_dir, detectors):
    """
    Compare multiple face detectors on dataset.
    """

    counters = {
        "RetinaFace": {"found": 0, "not_found": 0},
        "MTCNN": {"found": 0, "not_found": 0},
        "Dlib": {"found": 0, "not_found": 0},
        "OpenCV": {"found": 0, "not_found": 0}
    }

    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for filename in image_files:

        path = os.path.join(input_dir, filename)
        img = cv2.imread(path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # RetinaFace
        retina_count = len(detectors["retina"].get(img_rgb))
        counters["RetinaFace"]["found" if retina_count > 0 else "not_found"] += 1

        # MTCNN
        boxes, _ = detectors["mtcnn"].detect(img_rgb)
        mtcnn_count = 0 if boxes is None else len(boxes)
        counters["MTCNN"]["found" if mtcnn_count > 0 else "not_found"] += 1

        # Dlib
        dlib_count = len(detectors["dlib"](img_rgb, 1))
        counters["Dlib"]["found" if dlib_count > 0 else "not_found"] += 1

        # OpenCV
        opencv_count = len(detectors["opencv"].detectMultiScale(img_gray, 1.1, 5))
        counters["OpenCV"]["found" if opencv_count > 0 else "not_found"] += 1

    return counters