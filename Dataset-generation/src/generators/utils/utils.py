import glob
import cv2
import os


def get_filenames(path, regexp='*', extensions=None, recursive=True):
    if recursive:
        path += '**/'
    path += regexp

    files_grabbed = []
    for extension in extensions:
        files_grabbed.extend(glob.glob(path + '.' + extension, recursive=recursive))

    return files_grabbed

def load_image(file_path):
    return cv2.imread(file_path)

def save_image(file_path, image):
    cv2.imwrite(filename=file_path, img=image)

def leading_zeros(num, num_zeros):
        """
        Функция добавляет столько ведущих нулей, сколько указано в num_zeros,
        к строковому представлению числа num
        """
        num_str = str(num)
        num_zeros_to_add = num_zeros - len(num_str)  # сколько нулей нужно добавить
        return '0' * num_zeros_to_add + num_str  # добавляем нули и возвращаем строку

def count_digits(num):
    """
    Функция считает количество разрядов в числе num
    """
    if num == 0:
        return 1  # для числа 0 считаем, что у него один разряд
    count = 0
    while num > 0:
        count += 1
        num //= 10  # удаляем последнюю цифру числа
    return count


class Saver():
    def __init__(self, base_dir, files_per_dir=100, write_info_file=True, info_filename='_info.csv'):
        self.write_info_file = write_info_file
        self.base_dir = base_dir
        self.files_per_dir = files_per_dir

        if self.write_info_file:
            self.info_file = open(os.path.join(base_dir, info_filename), 'w')
            self.info_file.write('index\timage\tlabel\n')
        self.saved = 0
        self.num_zeros = count_digits(files_per_dir - 1)

    def save_images(self, images, one_image_by_item=True, subdirs=False, suffixes=None, extension='png'):
        """
        Функция сохранения N файлов в директории с динамическим созданием поддиректорий.

        :param base_dir: базовая директория для сохранения файлов.
        :param num_files: количество файлов для сохранения.
        :param files_per_dir: максимальное количество файлов в каждой поддиректории.
        """

        for i, item in enumerate(images):
            # Вычисляем, в какой директории должен быть сохранён файл
            n = i + self.saved

            dir_num = n // self.files_per_dir
            n_in_dir = n % self.files_per_dir

            dir_num_str = leading_zeros(dir_num, self.num_zeros)
            n_in_dir_str = leading_zeros(n_in_dir, self.num_zeros)

            # Создаём необходимую поддиректорию, если она ещё не создана
            item_path = os.path.join(self.base_dir, f"{dir_num_str}")
            os.makedirs(item_path, exist_ok=True)

            if subdirs:
                # Создаём поддиркеторию для item
                os.makedirs(os.path.join(item_path, f"{n_in_dir_str}"), exist_ok=True)

            if one_image_by_item:
                item = [item]

            info_str = f'{n}'

            for i, image in enumerate(item):
                if suffixes is not None:
                    suffix = suffixes[i]
                else: 
                    suffix = str(i)

                file_name = f"{dir_num_str}_{n_in_dir_str}_{suffix}.{extension}"

                # Формируем путь к новому файлу
                related_path = file_name
                if subdirs:
                    related_path = os.path.join(f"{n_in_dir_str}", file_name)
                related_path = os.path.join(f"{dir_num_str}", related_path)
                    
                file_path = os.path.join(self.base_dir, related_path)

                # Сохраняем файл
                save_image(file_path, image)
        
                info_str += f'\t{related_path}'
            
            info_str += '\n'
            if self.write_info_file:
                self.info_file.write(info_str)

        self.saved += len(images)
