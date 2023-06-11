from .base_augmenter import BaseAugmenter
from .cv_methods import resize


class EmptyAugmenter(BaseAugmenter):
    def __init__(self) -> None:
        super().__init__()
    
    def transform(self, image, max_size=None):
        return resize(image, max_size=max_size)