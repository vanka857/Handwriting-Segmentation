from .base_generator import BaseDatasetGenerator
from .base_augmenter import BaseAugmenter
from .empry_augmenter import EmptyAugmenter
from .utils import get_filenames, load_image, Saver, save_image
import cv2
import random
import numpy as np
import os
from tqdm import tqdm
import numpy as np


class HWRFromBoxesDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, printed_path, printed_replica_factor, boxes_path, result_path, printed_threshold=100, handwritten_threshold=100, boxes_per_paper=30, boxes_augmenter=None):
        configuration = {'printed_path': printed_path,
                         'boxes_path': boxes_path,
                         'result_path': result_path}
        super().__init__(configuration)
        
        self.printed_replica_factor = printed_replica_factor
        self.printed_threshold = printed_threshold
        self.boxes_per_paper = boxes_per_paper
        self.boxes_augmenter = boxes_augmenter if boxes_augmenter is not None else EmptyAugmenter()
        self.handwritten_threshold = handwritten_threshold
                         
        self.printed_files = get_filenames(self.configuration['printed_path'],
                                               extensions=['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'])    
        
        os.makedirs(result_path, exist_ok=True)

    def cover_printed_with_boxes(self, printed_image, get_box_func, dilate=True):
        printed_grey = cv2.cvtColor(printed_image, cv2.COLOR_BGR2GRAY)
        printed_inverse = cv2.bitwise_not(printed_grey)
        printed_mask = cv2.threshold(printed_inverse, 255 - self.handwritten_threshold, 255, cv2.THRESH_BINARY)[1]

        handwritten_mask = np.zeros_like(printed_grey)
        overlapped_image = printed_inverse

        for _ in range(self.boxes_per_paper):
            box = get_box_func(self)
            box_grey = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)

            box_grey = self.boxes_augmenter.transform(box_grey, max_size=overlapped_image.shape[:2])
            
            # print(type(box_grey))
            # print(box_grey)

            # print(box_grey)

            # cv2.imshow(box_grey)
            # cv2.waitKey()
            # return

            box_inverse = cv2.bitwise_not(box_grey)
            box_mask = cv2.threshold(box_inverse, 255 - self.handwritten_threshold, 255, cv2.THRESH_BINARY)[1]
            if dilate:
                box_mask = cv2.dilate(box_mask, kernel=np.ones((3, 3), np.uint8), iterations=1, borderType=cv2.BORDER_ISOLATED)

            box_masked = cv2.bitwise_and(box_inverse, box_mask)

            h, w = overlapped_image.shape
            h_box, w_box = box_masked.shape     
            # генерируем случайные координаты для наложения маленького изображения
            x = random.randint(0, w - w_box)
            y = random.randint(0, h - h_box)

            overlapped_image[y:y+h_box, x:x+w_box] = cv2.max(box_masked, overlapped_image[y:y+h_box, x:x+w_box])
            handwritten_mask[y:y+h_box, x:x+w_box] = cv2.bitwise_or(box_mask, handwritten_mask[y:y+h_box, x:x+w_box])

        mask = cv2.merge([np.zeros_like(handwritten_mask), printed_mask, handwritten_mask])

        return cv2.bitwise_not(overlapped_image), mask

    def create_dataset(self, boxes_map_file=None, paramsTuning=False):
        generated_count = 0
        self.result_saver = Saver(base_dir=self.configuration['result_path'])

        boxes_files = []
        if boxes_map_file is None:
            boxes_files = get_filenames(self.configuration['boxes_path'], 
                                           extensions=['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'])
        else:
            raise NotImplementedError
            with open(boxes_map_file, 'r') as f:
                for line in f:
                    col1 = line.strip().split('\t')[0]
                    boxes_files += [col1]
        
        if len(boxes_files) == 0:
            raise RuntimeError('There is no boxes_files or I can not read it!')
        
        self.boxes_files = boxes_files
        
        def get_box_func(self, idx=None):
            if idx is None:
                n = len(boxes_files)
                idx = random.randint(0, n - 1)
            return load_image(self.boxes_files[idx])
        
        # def get_printed_func(self, idx=None):
        #     if idx is None:
        #         n = len(self.printed_files)
        #         idx = random.randint(0, n - 1)
        #     return load_image(self.printed_files[idx])

        # while generated_count < count:
        #     printed_image = get_printed_func()

        # случайное наложение боксов на рукописные картинки 
        for printed_file in self.printed_files:
            for _ in range(self.printed_replica_factor):
                overlapped_image, handwritten_mask = self.cover_printed_with_boxes(printed_image=load_image(printed_file), get_box_func=get_box_func)
            
                # cv2.imshow('overlapped', overlapped_image)
                # cv2.waitKey()
                # cv2.imshow('masked', handwritten_mask)
                # cv2.waitKey()
            
                generated_count += 1

                if paramsTuning:
                    return overlapped_image, handwritten_mask
                else:
                    self.result_saver.save_images(images=[[overlapped_image, handwritten_mask]], one_image_by_item=False, subdirs=True, suffixes=['image', 'label'])
        
        return generated_count


class HWRFromImagesDatsetGenerator(HWRFromBoxesDatasetGenerator):
    def __init__(self, printed_path, printed_replica_factor, boxes_path, result_path, handwritten_path, handwritten_padding=[0, 0, 0, 0], handwritten_threshold=100, boxes_per_paper=30, boxes_augmenter=None):
        super().__init__(printed_path=printed_path, printed_replica_factor=printed_replica_factor, boxes_path=boxes_path, result_path=result_path, boxes_per_paper=boxes_per_paper, boxes_augmenter=boxes_augmenter)
        
        self.configuration['handwritten_path'] = handwritten_path
        self.handwritten_padding = handwritten_padding 
        self.handwritten_threshold = handwritten_threshold 
        
        os.makedirs(boxes_path, exist_ok=True)
        
        self.handwritten_files = get_filenames(self.configuration['handwritten_path'],
                                               extensions=['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'])

        self.boxes_created = False    

    def get_one_handwritten(self, idx):
        file = self.handwritten_files[idx]
        return load_image(file)

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
        h, w, _ = img.shape
        b_p, d_p, l_p, r_p = self.handwritten_padding
        img1 = img[int(b_p * h): int((1 - d_p) * h), int(l_p * w): int((1 - r_p) * w)]

        grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(grey, self.handwritten_threshold, 255, cv2.THRESH_BINARY)
        img_erode = cv2.erode(thresh, np.ones((10, 40), np.uint8), iterations=1)
        border = 15
        img_erode = cv2.copyMakeBorder(img_erode, border, border, border, border, cv2.BORDER_CONSTANT, value=(255))
        img1 = cv2.copyMakeBorder(img1, border, border, border, border, cv2.BORDER_REFLECT)

        # Выделение контуров 
        contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        output1 = img1.copy()

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

    def create_boxes(self):
        self.box_saver = Saver(base_dir=self.configuration['boxes_path'], write_info_file=False)

        for hwr_file in tqdm(self.handwritten_files, desc='Handwritten image processed: '):
            hwr_image = load_image(hwr_file)
            self.create_save_boxes(hwr_image)
        
        self.boxes_created = True

    def create_dataset(self, paramsTuning=False):
        if not self.boxes_created:
            self.create_boxes()

        return super(HWRFromImagesDatsetGenerator, self).create_dataset(paramsTuning=paramsTuning)
