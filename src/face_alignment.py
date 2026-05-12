from typing import Tuple
import numpy as np
from PIL import Image
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)


Point = Tuple[float, float]


def align_face_by_eyes_nose(
    image: np.ndarray,
    left_eye: Point,
    right_eye: Point,
    nose: Point
) -> tuple[np.ndarray, int]:
    """
    Align a face image based on eye and nose positions.

    The function determines the required rotation angle
    according to the relative position of the eyes and nose.

    Args:
        image (np.ndarray):
            Input face image.

        left_eye (Point):
            (x, y) coordinates of the left eye.

        right_eye (Point):
            (x, y) coordinates of the right eye.

        nose (Point):
            (x, y) coordinates of the nose.

    Returns:
        tuple[np.ndarray, int]:
            Rotated image and applied rotation angle.
    """

    # Midpoint between eyes
    forehead_center = (
        (left_eye[0] + right_eye[0]) / 2,
        (left_eye[1] + right_eye[1]) / 2
    )

    # Relative position
    dx = forehead_center[0] - nose[0]
    dy = forehead_center[1] - nose[1]

    # Determine rotation angle
    if abs(dx) > abs(dy):

        if dx > 0:
            rotation_angle = 90
            logging.info(
                "Rotating 90°: eyes are right of nose"
            )

        else:
            rotation_angle = -90
            logging.info(
                "Rotating -90°: eyes are left of nose"
            )

    else:

        if dy > 0:
            rotation_angle = 180
            logging.info(
                "Rotating 180°: eyes are below nose"
            )

        else:
            rotation_angle = 0
            logging.info(
                "No rotation needed"
            )

    # Rotate image
    image_pil = Image.fromarray(image)

    rotated_image = image_pil.rotate(
        rotation_angle,
        resample=Image.BICUBIC,
        expand=True
    )

    return np.array(rotated_image), rotation_angle