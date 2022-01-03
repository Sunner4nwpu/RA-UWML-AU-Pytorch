
import os
from PIL import Image
import cv2
import json
import argparse
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DISFADataset(data.Dataset):
    def __init__(self, opts, mode='train', split='CCNN'):
        self.debug = 1 if opts.snapshot == 'debug' else 0
        self.mode = mode
        self.split = split

        self.load_data_json(opts.json_dir, opts.json_name)
        self.get_train_valid(self.split)
        self.set_transform()

    def load_data_json(self, json_dir, json_name):

        data_json_path = os.path.join(json_dir, json_name)
        with open(data_json_path, 'r') as f:
            dataset_json = json.load(f)

        data_path = os.path.join(json_dir, dataset_json['image_path'])
        label_path = os.path.join(json_dir, dataset_json['label_path'])
        success_path = os.path.join(json_dir, dataset_json['success_path'])

        self.data = np.load(data_path, mmap_mode='r')
        self.label = np.load(label_path, mmap_mode='r')
        self.success = np.load(success_path, mmap_mode='r')

        self.mean = dataset_json['mean']
        self.std = dataset_json['std']
        self.au_number = self.label.shape[-1]

        label_info = dataset_json['label_info']
        subjects = len(label_info['subjects'])
        frames = label_info['frames']

        subjects_start_end = [0]
        for i in range(subjects):
            frame_start = subjects_start_end[i]
            frame_num = frames[i]
            subjects_start_end.append(frame_start + frame_num)

        self.subject_frames = []
        for i in range(subjects):
            frame_start = subjects_start_end[i]
            frame_end = subjects_start_end[i+1]
            frame_index = list(range(frame_start, frame_end))
            self.subject_frames.append(frame_index)

        print('Disfa Dataset step1: load data ok ...')


    def set_transform(self):
        if self.debug:
            normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        else:
            normalize = transforms.Normalize(self.mean, self.std)

        if self.mode == 'train':
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        elif self.mode == 'valid':
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
        else:
            raise ValueError('No {} dataset ...'.format(mode))

        print('Disfa Dataset step3: set transform ok ...')


    def get_train_valid(self, split):
        id2index = dict()
        id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        for i, id in enumerate(id_list):
            id2index['{}'.format(id)] = i

        if split == 'fold1':
            train_list = [2, 10, 1, 26, 27, 32, 30, 9, 16, 13, 18, 11, 28, 12, 6, 31, 21, 24]
            valid_list = [3, 29, 23, 25, 8, 5, 7, 17, 4]
        elif split == 'fold2':
            train_list = [2, 10, 1, 26, 27, 32, 30, 9, 16, 3, 29, 23, 25, 8, 5, 7, 17, 4]
            valid_list = [13, 18, 11, 28, 12, 6, 31, 21, 24]
        elif split == 'fold3':
            train_list = [13, 18, 11, 28, 12, 6, 31, 21, 24, 3, 29, 23, 25, 8, 5, 7, 17, 4]
            valid_list = [2, 10, 1, 26, 27, 32, 30, 9, 16]
        elif split == 'CCNN':
            train_list = [1, 5, 8, 9, 10, 11, 17, 18, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            valid_list = [2, 3, 4, 6, 7, 12, 13, 16, 23]       
        else:
            print('no split method ...')

        self.train_video = [int(id2index['{}'.format(id)]) for id in train_list]
        self.valid_video = [int(id2index['{}'.format(id)]) for id in valid_list]

        train_frames = []
        for video_index in self.train_video:
            train_frames += self.subject_frames[video_index]
        
        valid_frames = []
        for video_index in self.valid_video:
            valid_frames += self.subject_frames[video_index]

        if self.mode == 'train':
            self.data = self.data[train_frames]
            self.label = self.label[train_frames]
            self.success = self.success[train_frames]
            self.img_num = len(train_frames)
        else:
            self.data = self.data[valid_frames]
            self.label = self.label[valid_frames]
            self.success = self.success[valid_frames]
            self.img_num = len(valid_frames)      

        print('Disfa Dataset step2: set train valid ok ...')


    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transforms is not None:
            img = self.transforms(img)

        label = self.label[index]
        success = self.success[index]

        return img, label, success


    def __len__(self):
        return self.img_num


class AUBalanceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, use_sampler):
        print('initial balance sampler {}...'.format(use_sampler))

        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)

        au_label_count = {}
        for au in range(dataset.au_number):
            au_label_count[au] = dict()
            for au_value in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
                au_label_count[au][au_value] = 0

        for idx in self.indices:
            label = self._get_label(dataset, idx)
            for au in range(dataset.au_number):
                au_label_count[au][label[au]] += 1

        self.weights = torch.zeros(dataset.au_number, self.num_samples)
        for au in range(dataset.au_number):
            for sample in self.indices:
                label_sample = self._get_label(dataset, sample)
                label_sample_au = label_sample[au]
                self.weights[au, sample] = 1.0 / au_label_count[au][label_sample_au]

        if use_sampler == 1:
            self.au_weight = torch.FloatTensor([1. / dataset.au_number] * dataset.au_number)
        elif use_sampler == 2:
            self.au_weight = torch.FloatTensor([self.num_samples / v[1.0] for k, v in au_label_count.items()])
            

    def _get_label(self, dataset, idx):
        return dataset.label[idx]

    def __iter__(self):
        k = torch.multinomial(self.au_weight, 1).item()
        return (self.indices[i] for i in torch.multinomial(self.weights[k], self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples



def get_data_loader(opts, mode='train', split='CCNN'):
    dataset = DISFADataset(opts, mode, split)

    if mode == 'train':
        batch_size = opts.batch_size_train     
        sampler = None   
        shuffle = True
    else:
        batch_size = opts.batch_size_valid
        sampler = None
        shuffle = False

    if mode == 'train' and opts.use_sampler:
        sampler = AUBalanceSampler(dataset, opts.use_sampler)
        shuffle = False

    if opts.snapshot == 'debug':
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler
        )
    else:
        if device.type == 'cuda':
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                num_workers=16, pin_memory=True
            )
        else:
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler
            )

    return data_loader, len(dataset)

