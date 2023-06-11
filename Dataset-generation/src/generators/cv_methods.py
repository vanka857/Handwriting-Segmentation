import math
import cv2
import math


def resize(image, max_size=None, scale=1):
    h, w = image.shape[:2]

    if max_size is not None:
        max_scale = min(max_size[0], max_size[1]) / int(math.hypot(h, w) + 1)
        scale = min(scale, max_scale)
        
    return cv2.resize(image, dsize=(max(1, int(scale * w)), max(1, int(scale * h))), interpolation = cv2.INTER_CUBIC)

def rotate(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return outImg