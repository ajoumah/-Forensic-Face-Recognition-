import numpy as np


def crop_face(img, bbox, img_w, img_h):
    """
    Safely crop face from image using bounding box.
    """

    x1, y1, x2, y2 = bbox.astype(int)

    x1 = np.clip(x1, 0, img_w)
    y1 = np.clip(y1, 0, img_h)
    x2 = np.clip(x2, 0, img_w)
    y2 = np.clip(y2, 0, img_h)

    if x1 >= x2 or y1 >= y2:
        return None

    return img[y1:y2, x1:x2]