from .base_generator import BaseDatasetGenerator
from .empry_augmenter import EmptyAugmenter
from .utils import get_filenames, load_image, Saver
import cv2
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class FakeDataset(Dataset):
    def __init__(self, len, image_height, image_width, constant_value):
        self.len = len
        self.image_height = image_height
        self.image_width = image_width
        self.constant_value = constant_value
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return np.full((self.image_height, self.image_width, len(self.constant_value)), self.constant_value, np.uint8)


class ImageDataset(Dataset):
    def __init__(self, path, read_in_memory=False, transform=None, images_map_file=None, multiplication_factor=1) -> None:
        super().__init__()
        
        self.read_in_memory = read_in_memory
        self.transform = transform
        self.multiplication_factor = multiplication_factor

        self.images_files = []
        if images_map_file is None:
            self.images_files = get_filenames(path, 
                                        extensions=['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'])
        else:
            raise NotImplementedError
            with open(images_map_file, 'r') as f:
                for line in f:
                    col1 = line.strip().split('\t')[0]
                    self.images_files += [col1]
        
        if len(self.images_files) == 0:
            raise RuntimeError(f'There is no images in {path} or I can not read it!')
        
        self.raw_len = len(self.images_files)

        self.memory_images = []
        if read_in_memory:
            for image_file in tqdm(self.images_files, desc='Boxes read'):
                self.memory_images.append(load_image(image_file))
            self.raw_len = len(self.memory_images)


    def __len__(self) -> int:
        return self.raw_len * self.multiplication_factor

    def __getitem__(self, index):
        raw_index = index % self.raw_len
        image = None
        if self.read_in_memory:
            image = self.memory_images[raw_index]
        else:
            image = load_image(self.images_files[raw_index])

        if self.transform:
            image = self.transform(image)

        return image


class BoxDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, 
                 hwr_dataloader: DataLoader, 
                 boxes_path, 
                 hwr_threshold=100, 
                 write_info_file=False):
        super().__init__()
        self.hwr_dataloader = hwr_dataloader
        self.boxes_path = boxes_path
        self.hwr_threshold = hwr_threshold

        self.box_saver = Saver(base_dir=self.boxes_path, write_info_file=write_info_file)
    
    def create_save_boxes(self, img):
        '''
        Далее происходит:
        1. Обрезка изображения по заданной рамке (характерна для датасета)
        2. Преобразование в градации серого
        3. Бинаризация с подобранным вручную порогом (характерен для датасета)
        4. Размытие изображения с применением структурирующего элемента (Erode) c характерным для датасета размером ядра
        5. Добавление белой рамки с полученному и исходному изображению
        6. Поиск контуров для создания "рамок с текстом"
        7. Сохранение каждого найденного фрагмента текста в специальную директорию
        '''
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(grey, self.hwr_threshold, 255, cv2.THRESH_BINARY)
        img_erode = cv2.erode(thresh, np.ones((10, 40), np.uint8), iterations=1)
        border = 15
        img_erode = cv2.copyMakeBorder(img_erode, border, border, border, border, cv2.BORDER_CONSTANT, value=(255))
        img = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_REFLECT)

        # Выделение контуров 
        contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        output1 = img # img1.copy()

        areas = []
        rects = []

        for idx, contour in enumerate(contours):
            rect = (x, y, w, h) = cv2.boundingRect(contour)
            if hierarchy[0][idx][3] == 0:
                rects += [rect]
                areas += [w * h]

        boxes = []

        # Возьмём только контуры с площадью внутри квантили 20% - 90%
        indices = np.array(areas).argsort()
        l = len(indices)
        indices_cropped = indices[int(0.4 * l): int(0.9 * l)]

        for idx in indices_cropped:
            (x, y, w, h) = rects[idx]
            box = output1[y:y+h, x:x+w].copy()
            boxes += [box]

            # Для визуализации можно нарисовать контуры, но мы этого делать не будем
            # cv2.rectangle(output1, (x, y), (x + w, y + h), (255, 0, 0), 2)

        self.box_saver.save_images(images=boxes)
        return 0

    def create_dataset(self):
        for hwr_sample in tqdm(self.hwr_dataloader, desc='Handwritten image processed: '):
            for hwr in hwr_sample:
                self.create_save_boxes(hwr.numpy())


class MixDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, 
                 printed_dataloader: DataLoader, 
                 printed_replica_factor, 
                 boxes_dataloader: DataLoader, 
                 result_path, 
                 boxes_per_printed=100,
                 printed_threshold=100, 
                 hwr_threshold=100, 
                 boxes_augmenter=None,
                 write_info_file=True):
        super().__init__()

        self.printed_dataloader = printed_dataloader
        self.printed_replica_factor = printed_replica_factor
        # self.boxes_dataloader = boxes_dataloader
        self.boxes_iterable = iter(boxes_dataloader)
        self.boxes_per_printed = boxes_per_printed
        self.result_path = result_path        
        self.printed_threshold = printed_threshold
        self.hwr_threshold = hwr_threshold
        self.boxes_augmenter = boxes_augmenter if boxes_augmenter is not None else EmptyAugmenter()
                            
        self.result_saver = Saver(base_dir=result_path, write_info_file=write_info_file)

    def cover_printed_with_boxes(self, printed_image, dilate=True):
        printed_grey = cv2.cvtColor(printed_image, cv2.COLOR_BGR2GRAY)
        printed_inverse = cv2.bitwise_not(printed_grey)
        printed_mask = cv2.threshold(printed_inverse, 255 - self.hwr_threshold, 255, cv2.THRESH_BINARY)[1]

        handwritten_mask = np.zeros_like(printed_grey)
        overlapped_image = printed_inverse

        for i, box in enumerate(self.boxes_iterable):
            box = box[0].numpy() # iter(self.boxes_dataloader).next().numpy()[0]
            box_grey = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
            box_grey = self.boxes_augmenter.transform(box_grey, max_size=overlapped_image.shape[:2])

            box_inverse = cv2.bitwise_not(box_grey)
            box_mask = cv2.threshold(box_inverse, 255 - self.hwr_threshold, 255, cv2.THRESH_BINARY)[1]
            if dilate:
                box_mask = cv2.dilate(box_mask, kernel=np.ones((3, 3), np.uint8), iterations=1, borderType=cv2.BORDER_ISOLATED)

            box_masked = cv2.bitwise_and(box_inverse, box_mask)

            # генерируем случайные координаты для наложения маленького изображения
            h, w = overlapped_image.shape
            h_box, w_box = box_masked.shape     
            x = random.randint(0, w - w_box)
            y = random.randint(0, h - h_box)

            # наложение через попиксельный максимум
            overlapped_image[y:y+h_box, x:x+w_box] = cv2.max(box_masked, overlapped_image[y:y+h_box, x:x+w_box])
            handwritten_mask[y:y+h_box, x:x+w_box] = cv2.bitwise_or(box_mask, handwritten_mask[y:y+h_box, x:x+w_box])

            if i == self.boxes_per_printed - 1:
                break

        mask = cv2.merge([np.zeros_like(handwritten_mask), printed_mask, handwritten_mask])

        return cv2.bitwise_not(overlapped_image), mask

    def create_dataset(self, paramsTuning=False):
        generated_count = 0
        
        for printed_sample in tqdm(self.printed_dataloader, desc='Printed images batch processed: '):
            for printed in printed_sample:
                printed = printed.numpy()
                for _ in range(self.printed_replica_factor):
                    overlapped_image, handwritten_mask = self.cover_printed_with_boxes(printed_image=printed)

                    # cv2.imshow('overlapped', overlapped_image)
                    # cv2.waitKey()
                    # cv2.imshow('masked', handwritten_mask)
                    # cv2.waitKey()
                
                    generated_count += 1

                    if paramsTuning:
                        return overlapped_image, handwritten_mask
                    else:
                        self.result_saver.save_images(images=[[overlapped_image, handwritten_mask]], 
                                                    one_image_by_item=False, 
                                                    subdirs=True, 
                                                    suffixes=['image', 'label'])
        
        return generated_count
