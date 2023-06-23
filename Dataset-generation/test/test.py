from src.generators import BoxDatasetGenerator, ImageDataset, MixDatasetGenerator, MyAugmenter
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import os


path = '/Users/vankudr/Yandex.Disk.localized/MIPT/НИР/Handwriting-Segmentation/Dataset-generation/data/example-dataset-2023'

# # CREATING BOX DATASET FROM HANDWRITTEN
# hwr_path = os.path.join(path, 'input/hwr/')
# boxes_path = os.path.join(path, 'input/boxes/')

# def pad(img, padding):
#     h, w, _ = img.shape
#     b_p, d_p, l_p, r_p = padding
#     return img[int(b_p * h): int((1 - d_p) * h), int(l_p * w): int((1 - r_p) * w)]

# hwr_transformer = lambda img: pad(img=img, padding=[0.23, 0.25, 0.01, 0.01])
# hwr_dataset = ImageDataset(path=hwr_path, transform=hwr_transformer)
# hwr_dataloader = DataLoader(hwr_dataset, batch_size=1, shuffle=True)

# box_gen = BoxDatasetGenerator(hwr_dataloader=hwr_dataloader,
#                               boxes_path=boxes_path,
#                               hwr_threshold=160)
# box_gen.create_dataset()


# CREATING MIX DATASET FROM BOXES AND PRINTED
printed_path = os.path.join(path, 'input/printed/')
printed_dataset = ImageDataset(path=printed_path)

boxes_path = os.path.join(path, 'input/boxes/')
boxes_dataset = ImageDataset(path=boxes_path, read_in_memory=True, multiplication_factor=100)

result_path = os.path.join(path, 'output/result3/')

def create_mix_dataset(printed_dataset, boxes_dataset, result_path):
    printed_dataloader = DataLoader(printed_dataset, batch_size=1, shuffle=True)
    boxes_dataloader = DataLoader(boxes_dataset, batch_size=1, shuffle=True)
    boxes_aug = MyAugmenter(random_scaling=[0.15, 0.4], random_rotation=[-5, 5])
    mix_gen = MixDatasetGenerator(printed_dataloader=printed_dataloader,
                                printed_replica_factor=10,
                                boxes_dataloader=boxes_dataloader,
                                boxes_per_printed=300,
                                result_path=result_path,
                                printed_threshold=100,
                                hwr_threshold=160,
                                boxes_augmenter=boxes_aug)

    mix_gen.create_dataset()

create_mix_dataset(printed_dataset, boxes_dataset, result_path)

# mix_data = {'train': (printed_train_dataset, boxes_train_dataset),
#             'test': (printed_test_dataset, boxes_test_dataset)}

# for key, item in mix_data.items:
#     print(f'Starting {key} generation')

