from .base_augmenter import BaseAugmenter
from .cv_methods import resize, rotate
import random
    

class MyAugmenter(BaseAugmenter):
    def __init__(self, random_rotation=None, random_scaling=None) -> None:
        super().__init__()
        self.random_rotation = random_rotation
        self.random_scaling = random_scaling

    def get_one_transformed(self, image, max_size=None):
        if self.random_scaling is not None:
            scale = random.uniform(self.random_scaling[0], self.random_scaling[1])
            
        image = resize(image, scale=scale, max_size=max_size)

        if self.random_rotation:
            angle = random.uniform(self.random_rotation[0], self.random_rotation[1])
            image = rotate(image, angle)
        
        return image

    def transform(self, image, max_size=None, mult_factor=1): # , yelding=False)
        # if yelding:
        #     print(1)
        #     for _ in range(mult_factor):
        #         yield image
        # else 
        if mult_factor == 1:
            return self.get_one_transformed(image, max_size)
        else:
            images = []
            for _ in range(mult_factor):
                images += [self.get_one_transformed(image, max_size)]
        