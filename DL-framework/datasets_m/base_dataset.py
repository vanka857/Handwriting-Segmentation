'''
This module implements an abstract base class (ABC) 'BaseDataset' for datasets. 
Also includes some transformation functions.
'''
from abc import ABC, abstractmethod
import torch.utils.data as data

class BaseDataset(data.Dataset, ABC):
    '''
    Это асбстрактный базовый класс для датасетов
    '''

    @abstractmethod
    def __len__(self):
        '''
        Return the total number of images in the dataset.
        '''
        return 0

    @abstractmethod
    def __getitem__(self, index):
        '''
        Return a data point (usually data and labels in
        a supervised setting).
        '''
        pass

    def pre_epoch_callback(self, epoch):
        '''
        Callback to be called before every epoch.
        '''
        pass

    def post_epoch_callback(self, epoch):
        '''
        Callback to be called after every epoch.
        '''
        pass