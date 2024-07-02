import os
import random

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from scipy import io
from PIL import ImageOps, Image
import torch
import torchvision.transforms as transforms
import h5py as h5
import numpy as np


resizer_collection = {"nearest": InterpolationMode.NEAREST,
                      "box": InterpolationMode.BOX,
                      "bilinear": InterpolationMode.BILINEAR,
                      "hamming": InterpolationMode.HAMMING,
                      "bicubic": InterpolationMode.BICUBIC,
                      "lanczos": InterpolationMode.LANCZOS}

class RandomCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        size = (min(img.size), min(img.size))
        # Only step forward along this edge if it's the long edge
        i = (0 if size[0] == img.size[0] else np.random.randint(low=0, high=img.size[0] - size[0]))
        j = (0 if size[1] == img.size[1] else np.random.randint(low=0, high=img.size[1] - size[1]))
        return transforms.functional.crop(img, j, i, size[0], size[1])

    def __repr__(self):
        return self.__class__.__name__


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img[0].shape))

    def __repr__(self):
        return self.__class__.__name__


class Astech_PN_Dataset(Dataset):
    def __init__(self,
                 data_dir,
                 train=True,
                 crop_long_edge=True,
                 resize_size=(32, 32),
                 resizer="bilinear",
                 random_flip=False,
                 normalize=True,
                 load_data_in_memory=False):
        super(Astech_PN_Dataset, self).__init__()
        self.data_dir = data_dir
        self.train = train
        self.random_flip = random_flip
        self.normalize = normalize
        self.load_data_in_memory = load_data_in_memory
        self.trsf_list = []
        self.class_names = ['0PN', '1PN', '2PN', 'mPN']
        self.class_ratio = [0.25, 0.25, 0.25, 0.25]
        self.collect_0PN_file_names()
        self.collect_1PN_file_names()
        self.collect_2PN_file_names()
        self.collect_mPN_file_names()


        self.trsf_list += [transforms.ToTensor()]
        if crop_long_edge:
            self.trsf_list += [CenterCropLongEdge()]

        if resize_size is not None and resizer != "wo_resize":
            self.trsf_list += [transforms.Resize(resize_size, interpolation=resizer_collection[resizer])]

        if self.random_flip:
            self.trsf_list += [transforms.RandomHorizontalFlip()]

        if self.normalize:
            self.trsf_list += [transforms.Normalize(mean=[0.5] * 11, std=[0.5] * 11)]

        self.trsf = transforms.Compose(self.trsf_list)


    def collect_0PN_file_names(self):
        """
        return frame (time) that belong to 0PN class
        """
        self.list_0PN_file_names = []

        class_folder_names = ['0PN']
        range_time_to_get = [16, 0] # moment (hour) that should to get frames
        class_folders = [os.path.join(self.data_dir, fn) for fn in class_folder_names]

        for class_fold in class_folders:
            well_folders = [os.path.join(class_fold, fn) for fn in os.listdir(class_fold)]

            for well_fold in well_folders:
                str_times = [fn.split('_')[0] for fn in os.listdir(well_fold)]
                times = [{'hour': int(time[1:3]), 'minute': int(time[3:5]), 'second': int(time[5:7])} for time in str_times]

                sorted_times = sorted(times, key=lambda x: x['hour'] * 3600 + x['minute'] * 60 + x['second'])

                for h in range_time_to_get:
                    h_ids = [i for i, time in enumerate(sorted_times) if h == time['hour']]
                    if len(h_ids) == 0:
                        continue
                    frame_idx = (h_ids[-1] + h_ids[0])//2
                    ch_time = sorted_times[frame_idx]
                    h = str(ch_time['hour']) if ch_time['hour'] > 9 else str(0) + str(ch_time['hour'])
                    m = str(ch_time['minute']) if ch_time['minute'] > 9 else str(0) + str(ch_time['minute'])
                    s = str(ch_time['second']) if ch_time['second'] > 9 else str(0) + str(ch_time['second'])
                    ch_fn = str(0) + h + m + s
                    self.list_0PN_file_names.append(os.path.join(well_fold, ch_fn))




    def collect_1PN_file_names(self):
        """
        return frame (time) that belong to 0PN class
        """
        self.list_1PN_file_names = []

        class_folder_names = ['1PN']
        range_time_to_get = [16] # moment (hour) that should to get frames
        class_folders = [os.path.join(self.data_dir, fn) for fn in class_folder_names]

        for class_fold in class_folders:
            well_folders = [os.path.join(class_fold, fn) for fn in os.listdir(class_fold)]

            for well_fold in well_folders:
                str_times = [fn.split('_')[0] for fn in os.listdir(well_fold)]

                times = [{'hour': int(time[1:3]), 'minute': int(time[3:5]), 'second': int(time[5:7])} for time in str_times]

                sorted_times = sorted(times, key=lambda x: x['hour'] * 3600 + x['minute'] * 60 + x['second'])

                for h in range_time_to_get:
                    h_ids = [i for i, time in enumerate(sorted_times) if h == time['hour']]
                    if len(h_ids) == 0:
                        continue
                    frame_idx = (h_ids[-1] + h_ids[0])//2
                    ch_time = sorted_times[frame_idx]
                    h = str(ch_time['hour']) if ch_time['hour'] > 9 else str(0) + str(ch_time['hour'])
                    m = str(ch_time['minute']) if ch_time['minute'] > 9 else str(0) + str(ch_time['minute'])
                    s = str(ch_time['second']) if ch_time['second'] > 9 else str(0) + str(ch_time['second'])
                    ch_fn = str(0) + h + m + s
                    self.list_1PN_file_names.append(os.path.join(well_fold, ch_fn))



    def collect_2PN_file_names(self):
        """
        return frame (time) that belong to 0PN class
        """
        self.list_2PN_file_names = []

        class_folder_names = ['1PN']
        range_time_to_get = [16] # moment (hour) that should to get frames
        class_folders = [os.path.join(self.data_dir, fn) for fn in class_folder_names]

        for class_fold in class_folders:
            well_folders = [os.path.join(class_fold, fn) for fn in os.listdir(class_fold)]

            for well_fold in well_folders:
                str_times = [fn.split('_')[0] for fn in os.listdir(well_fold)]

                times = [{'hour': int(time[1:3]), 'minute': int(time[3:5]), 'second': int(time[5:7])} for time in str_times]

                sorted_times = sorted(times, key=lambda x: x['hour'] * 3600 + x['minute'] * 60 + x['second'])

                for h in range_time_to_get:
                    h_ids = [i for i, time in enumerate(sorted_times) if h == time['hour']]
                    if len(h_ids) == 0:
                        continue
                    frame_idx = (h_ids[-1] + h_ids[0])//2
                    ch_time = sorted_times[frame_idx]
                    h = str(ch_time['hour']) if ch_time['hour'] > 9 else str(0) + str(ch_time['hour'])
                    m = str(ch_time['minute']) if ch_time['minute'] > 9 else str(0) + str(ch_time['minute'])
                    s = str(ch_time['second']) if ch_time['second'] > 9 else str(0) + str(ch_time['second'])
                    ch_fn = str(0) + h + m + s
                    self.list_2PN_file_names.append(os.path.join(well_fold, ch_fn))




    def collect_mPN_file_names(self):
        """
        return frame (time) that belong to 0PN class
        """
        self.list_mPN_file_names = []

        all_class_set = set(os.listdir(self.data_dir))
        clear_class_set = set(['0PN', '1PN', '2PN'])
        class_folder_names = list(all_class_set - clear_class_set)

        range_time_to_get = [16] # moment (hour) that should to get frames
        class_folders = [os.path.join(self.data_dir, fn) for fn in class_folder_names]

        for class_fold in class_folders:
            well_folders = [os.path.join(class_fold, fn) for fn in os.listdir(class_fold)]

            for well_fold in well_folders:
                str_times = [fn.split('_')[0] for fn in os.listdir(well_fold)]

                times = [{'hour': int(time[1:3]), 'minute': int(time[3:5]), 'second': int(time[5:7])} for time in str_times]

                sorted_times = sorted(times, key=lambda x: x['hour'] * 3600 + x['minute'] * 60 + x['second'])

                for h in range_time_to_get:
                    h_ids = [i for i, time in enumerate(sorted_times) if h == time['hour']]
                    if len(h_ids) == 0:
                        continue
                    frame_idx = (h_ids[-1] + h_ids[0])//2
                    ch_time = sorted_times[frame_idx]
                    h = str(ch_time['hour']) if ch_time['hour'] > 9 else str(0) + str(ch_time['hour'])
                    m = str(ch_time['minute']) if ch_time['minute'] > 9 else str(0) + str(ch_time['minute'])
                    s = str(ch_time['second']) if ch_time['second'] > 9 else str(0) + str(ch_time['second'])
                    ch_fn = str(0) + h + m + s
                    self.list_mPN_file_names.append(os.path.join(well_fold, ch_fn))



    def __getitem__(self, index):
        id2label = {i: label for i, label in enumerate(self.class_names)}
        class_id = random.choices(list(id2label.keys()), self.class_ratio, k=1)[0]
        class_name = id2label[class_id]

        if class_name == '0PN':
            frame = random.choice(self.list_0PN_file_names)
        elif class_name == '1PN':
            frame = random.choice(self.list_1PN_file_names)
        elif class_name == '2PN':
            frame = random.choice(self.list_2PN_file_names)
        elif class_name == 'mPN':
            frame = random.choice(self.list_mPN_file_names)

        frame_files = [frame + '_' + str(i) + '.jpg' for i in range(11)]
        stack_imgs = []
        for imf in frame_files:
            img = np.array(Image.open(imf).convert('L'))
            stack_imgs += [img]
        stack_11_img = np.stack(stack_imgs, axis=2)
        return self.trsf(stack_11_img), int(class_id)

    def __len__(self):
        if self.train == True:
            return 100
        else:
            return 40