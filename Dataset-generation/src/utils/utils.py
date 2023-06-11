import numpy as np
from PIL import Image


def image_to_array(image):
    return np.array(image)

def array_to_image(array):
    return Image.fromarray(np.array(array, dtype=np.uint8))