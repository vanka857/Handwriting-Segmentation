from abc import ABC, abstractmethod

class BaseDatasetGenerator(ABC):
    def __init__(self, configuration):
        self.configuration = configuration
        super().__init__()
        
    @abstractmethod
    def create_dataset(self, *args, **kwargs):
        raise NotImplementedError
