import cv2
from generators import HWRFromImagesDatsetGenerator, HWRFromBoxesDatasetGenerator, MyAugmenter

path = '/Users/vankudr/Yandex.Disk.localized/MIPT/НИР/Handwriting-Segmentation/Dataset-generation/data/example-dataset-2023'

box_aug = MyAugmenter(random_scaling=[0.2, 0.6], random_rotation=[-5, 5])
gen = HWRFromImagesDatsetGenerator(printed_path=path+'/input/printed/',
                                   printed_replica_factor=10,
                                   boxes_path=path+'/input/boxes/',
                                   result_path=path+'/output/result/',
                                   handwritten_path=path+'/input/hwr/', 
                                   handwritten_padding=[0.23, 0.25, 0.01, 0.01],
                                   handwritten_threshold=160,
                                   boxes_per_paper=300,
                                   boxes_augmenter=box_aug)

# gen.create_boxes()
gen.create_dataset()

# gen2 = HWRFromBoxesDatasetGenerator(printed_path=path+'/input/printed/',
#                                    printed_replica_factor=10,
#                                    boxes_path=path+'/input/boxes2/',
#                                    result_path=path+'/output/result2/',
#                                    boxes_per_paper=300,
#                                    boxes_augmenter=box_aug)

# i1, i2 = gen2.create_dataset(paramsTuning=True)

# cv2.imshow('1', i1)
# cv2.imshow('2', i2)
# cv2.waitKey()
