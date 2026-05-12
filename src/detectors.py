
import cv2
import dlib
import numpy as np
from facenet_pytorch import MTCNN
from insightface.app import FaceAnalysis


def load_detectors():
    """
    Initialize and return all face detectors.
    """

    retina = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    retina.prepare(ctx_id=0)

    mtcnn = MTCNN(keep_all=True, device="cpu")
    dlib_detector = dlib.get_frontal_face_detector()

    opencv_detector = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml"
    )

    return {
        "retina": retina,
        "mtcnn": mtcnn,
        "dlib": dlib_detector,
        "opencv": opencv_detector
    }