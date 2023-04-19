from abc import ABC,  abstractclassmethod


class BaseAugmenter(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractclassmethod
    def transform(self, image, mult_factor=1):
        raise NotImplementedError
