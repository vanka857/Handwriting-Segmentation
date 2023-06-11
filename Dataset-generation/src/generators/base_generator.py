from abc import ABC, abstractmethod

class BaseDatasetGenerator(ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def create_dataset(self, *args, **kwargs):
        raise NotImplementedError
