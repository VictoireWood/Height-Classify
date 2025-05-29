import pandas as pd
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from glob import glob
import numpy as np
from utils.utils import image_fft, tensor_to_image, tensor_fft_3D

import os
import parser
import random
import logging
from collections import defaultdict
import bisect
import platform

args = parser.parse_arguments()

basic_transform = T.Compose([
    T.Resize(args.test_resize, antialias=True),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_satellite_image_path(image_path):
    image_folder = os.path.dirname(image_path)  # 显示高度的文件夹路径
    class_folder = os.path.dirname(image_folder)
    base_folder = os.path.dirname(class_folder)
    base_folder_name = os.path.basename(base_folder)
    satellite_image_folder = class_folder.replace(base_folder_name, 'satellite-view')
    satellite_image_path = os.path.join(satellite_image_folder, '0.png')
    return satellite_image_path

def get__class_id__group_id(height, M, N):
    """Return class_id and group_id for a given point.
        The class_id is a tuple of UTM_east, UTM_north (e.g. (396520, 4983800)).
        The group_id represents the group to which the class belongs
        (e.g. (0, 1), and it is between (0, 0) and (N, N).
    """
    class_id = int(height // M * M)  # Rounded to nearest lower multiple of M

    # group_id goes from (0, 0) to (N, N)
    group_id = (class_id % (M * N) // M)
    return class_id, group_id

def get_heights_from_paths(images_paths: list[str]):
    def get_height_from_path(path):
        folder_path = os.path.dirname(path)
        folder_name = os.path.basename(folder_path)
        height = int(folder_name)
        return height

    heights = [get_height_from_path(image_path) for image_path in images_paths]    # NOTE /ctf53sc7v38s73e0mksg/maps/SUES-200-512x512/drone_view_512/0003/300/13.jpg
    return heights

def initialize(dataset_folder, dataset_name, M, N, min_images_per_class):
    paths_file = f"cache/paths_{dataset_name}_M{M}_N{N}_mipc{min_images_per_class}.torch"
    # Search paths of dataset only the first time, and save them in a cached file
    if not os.path.exists(paths_file):
        logging.info(f"Searching training images in {dataset_folder}")
        # images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg", recursive=True))   # ORIGION
        sub_folders = [f'{order:04d}' for order in range(1, 121)]
        images_paths = []
        for sub_folder in sub_folders:
            current_images_paths = sorted(glob(f"{dataset_folder}/{sub_folder}/**/*.jpg", recursive=True))  # SUES 的图像格式是jpg
            images_paths.extend(current_images_paths)
        # images_paths = sorted(glob(f"{dataset_folder}/**/*.png", recursive=True))   # EDIT
        # Remove folder_path from images_path, so that the same cache file can be used on any machine
        images_paths = [p.replace(dataset_folder, "") for p in images_paths]
        
        os.makedirs("cache", exist_ok=True)
        torch.save(images_paths, paths_file)
    else:
        images_paths = torch.load(paths_file)

    logging.info(f"Found {len(images_paths)} images")

    heights = get_heights_from_paths(images_paths)

    logging.info("For each image, get its flight height from its path")
    logging.info("For each image, get class and group to which it belongs")
    class_id__group_id = [get__class_id__group_id(h, M, N) for h in heights]

    logging.info("Group together images belonging to the same class")
    images_per_class = defaultdict(list)
    images_per_class_per_group = defaultdict(dict)
    for image_path, (class_id, _) in zip(images_paths, class_id__group_id):
        images_per_class[class_id].append(image_path)

    # Images_per_class is a dict where the key is class_id, and the value
    # is a list with the paths of images within that class.
    images_per_class = {k: v for k, v in images_per_class.items() if len(v) >= min_images_per_class}

    logging.info("Group together classes belonging to the same group")
    # Classes_per_group is a dict where the key is group_id, and the value
    # is a list with the class_ids belonging to that group.
    classes_per_group = defaultdict(set)
    for class_id, group_id in class_id__group_id:
        if class_id not in images_per_class:
            continue  # Skip classes with too few images
        classes_per_group[group_id].add(class_id)   # 每个group_id对应的class_id可以保证没有重复元素，格式：defaultdict(<class 'set'>, {0: {0,2,4}})

    for group_id, group_classes in classes_per_group.items():   # NOTE 这里得到的group_id是一个数，group_classes是一个set，格式：dict_items([(0, {0, 20}), (1, {1, 2})])，group_id和group_classes分别对应，0和{0, 20}，1和{1, 2}
        for class_id in group_classes:
            images_per_class_per_group[group_id][class_id] = images_per_class[class_id] # NOTE images_per_class_per_group 格式：defaultdict(<class 'dict'>, {0: {0: 'power.png'}, 1: {1: 'pwr.png'}})
    # Convert classes_per_group to a list of lists.
    # Each sublist represents the classes within a group.
    classes_per_group = [list(c) for c in classes_per_group.values()]   # classes_per_group.values()的格式：dict_values([{0, 20}, {1, 2}])；list(c)的格式：[0, 20], classes_per_group格式：[[0, 20], [1, 2]]
    classes_per_group= [sorted(sublist) for sublist in classes_per_group]   # group的sublist中，class的数值从小到大排序
    images_per_class_per_group = [c for c in images_per_class_per_group.values()]   # NOTE images_per_class_per_group 格式：[{0: ['power.png']}, {1: ['pwr.png']}, {2: ['1.png', '2.png'], 1: ['4.png', '5.png']}]
    images_per_class_per_group = [{k: v for k, v in sorted(subdict.items())} for subdict in images_per_class_per_group] # [{100: ['/ct02/0100/@ct02@100.00@00@0.00@0@0@.png', ...], 200: [...], 300: [...], 400: [...], 500: [...], 600: [...]}, {450: [...], 550: [...], 650: [...], 150: [...], 250: [...], 350: [...]}]

    images_num_per_class_per_group = []
    ends_per_group = []
    for group in range(len(images_per_class_per_group)):
        images_per_class_in_current_group = images_per_class_per_group[group]
        images_num_per_class_in_current_group = defaultdict(int)
        for class_id, images_paths_list in images_per_class_in_current_group.items():
            images_num_per_class_in_current_group[class_id] = len(images_paths_list)
        images_num_per_class_per_group.append(images_num_per_class_in_current_group)
        images_num_per_class_in_current_group_list = list(images_num_per_class_in_current_group.values())
        ends = [sum(images_num_per_class_in_current_group_list[:i+1]) for i in range(len(images_num_per_class_in_current_group))]

        ends_per_group.append(ends)

    return classes_per_group, images_per_class_per_group, ends_per_group


class SUES(Dataset):
    def __init__(self, group_num, dataset_name, train_path, M = args.M, N = args.N, min_images_per_class = 15,transform=basic_transform):
        super().__init__()

        cache_filename = f"cache/{dataset_name}_M{M}_N{N}_mipc{min_images_per_class}.torch"
        if not os.path.exists(cache_filename):
            classes_per_group, images_per_class_per_group, ends_per_group = initialize(train_path, dataset_name, M, N, min_images_per_class)
            torch.save((classes_per_group, images_per_class_per_group, ends_per_group), cache_filename)
        else:
            classes_per_group, images_per_class_per_group, ends_per_group = torch.load(cache_filename)
        classes_ids = classes_per_group[group_num]
        images_per_class = images_per_class_per_group[group_num]    # 当前group中所有class的图片相对train_path的路径

        ends = ends_per_group[group_num]

        # self.transform = transform
        # images_paths = sorted(glob(f"{base_path}/**/*.png", recursive=True))
        
        # self.images_paths = images_paths
        # self.heights = self.get_heights(images_paths)

        self.train_path = train_path

        self.M = M
        self.N = N
        self.transform = transform
        self.classes_ids = classes_ids  # classes_ids是一个list，格式：[0, 20]，代表当前group中的所有class的id
        self.images_per_class = images_per_class
        self.class_centers = self.classes_ids
        self.classes_num_total = len(classes_ids)

        self.ends = ends

        self.fft = args.fft
        self.fft_log_base = args.fft_log_base

        self.group_len = self.get_images_num()

    def __getitem__(self, index):

        class_num_current = bisect.bisect_right(self.ends, index)
        assert class_num_current < self.classes_num_total, 'class_num_current >= classes_num_total'
        # class_num_current = random.randint(0, self.classes_num_total - 1)
        class_id_current = self.classes_ids[class_num_current]
        class_center_current = self.class_centers[class_num_current]

        image_path = self.train_path + random.choice(self.images_per_class[class_id_current])
        
        try:
            image = Image.open(image_path).convert('RGB')
        except UnidentifiedImageError:
            logging.info(f"ERR: There was an error while reading image {image_path}, it is probably corrupted")
            image = torch.zeros([3, args.train_resize[0], args.train_resize[1]])
        
        if self.transform:
            image = self.transform(image)

        if self.fft:
            image = tensor_fft_3D(image, self.fft_log_base)
        
        return image, class_num_current, class_id_current
    
    def get_images_num(self):
        """Return the number of images within this group."""
        return sum([len(self.images_per_class[c]) for c in self.classes_ids])
    
    def get_classes_num(self):
        """Return the number of classes within this group."""
        return len(self.classes_ids)
    
    def __len__(self):
        """Return a large number. This is because if you return the number of
        classes and it is too small (like in pitts30k), the dataloader within
        InfiniteDataLoader is often recreated (and slows down training).
        """
        return self.group_len
        return 1000000

class SUESTest(torch.utils.data.Dataset):
    def __init__(self, test_folder, M=10, N=5, image_size=256):
        super().__init__()
        logging.debug(f"Searching test images in {test_folder}")

        # images_paths = sorted(glob(f"{test_folder}/**/*.jpg", recursive=True))    # ORIGION

        sub_folders = [f'{order:04d}' for order in range(121, 201)]
        images_paths = []
        for sub_folder in sub_folders:
            current_images_paths = sorted(glob(f"{test_folder}/{sub_folder}/**/*.jpg", recursive=True))
            images_paths.extend(current_images_paths)
        # images_paths = sorted(glob(f"{test_folder}/**/*.png", recursive=True))   # EDIT

        logging.debug(f"Found {len(images_paths)} images")

        self.heights = get_heights_from_paths(images_paths)

        class_id_group_id = [get__class_id__group_id(h, M, N) for h in self.heights]    # 得到(class_id, group_id)
        self.images_paths = images_paths
        self.class_centers = [id[0] for id in class_id_group_id]
        self.class_id = [id[0] for id in class_id_group_id]
        self.group_id = [id[1] for id in class_id_group_id]

        self.normalize = T.Compose([
            T.Resize(image_size, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.fft = args.fft
        self.fft_log_base = args.fft_log_base

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        # class_id = self.class_id[index]

        pil_image = Image.open(image_path).convert('RGB')
        # pil_image = T.functional.resize(pil_image, self.shapes[index])
        image = self.normalize(pil_image)

        if self.fft:
            image = tensor_fft_3D(image, self.fft_log_base)
        
        # if isinstance(image, tuple):
        #     image = torch.stack(image, dim=0)
        return image, self.class_id[index], self.heights[index], image_path

    def __len__(self):
        return len(self.images_paths)
    
class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch