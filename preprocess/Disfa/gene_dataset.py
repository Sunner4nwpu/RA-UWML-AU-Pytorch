
import os
import csv
import json
import argparse

from tqdm import tqdm
import numpy as np
from PIL import Image
from numpy.lib.format import open_memmap



def process_labels(label_dir):
    label_info = dict()

    subjects = sorted(os.listdir(label_dir))
    label_info['subjects'] = subjects

    label_info['frames'] = []
    labels = np.empty((0, 12)).astype(np.float32)    

    for subject in subjects:
        sub_dir = os.path.join(label_dir, subject)
        sub_txts = sorted(os.listdir(sub_dir))
        sub_labels = []
        sub_len = []

        for each_txt in sub_txts:
            txt_data = []
            txt_path = os.path.join(sub_dir, each_txt)
            with open(txt_path, 'r') as f:
                all_line = f.readlines()
                for each_line in all_line:
                    txt_data.append(float(each_line.split(',')[-1])) 
            sub_len.append(len(txt_data))           
            sub_labels.append(np.array(txt_data))

        for i in range(len(sub_len)):
            assert sub_len[0] == sub_len[i], 'error label files ...'
        label_info['frames'].append(sub_len[0])

        subject_label = np.stack(sub_labels, axis=1)
        labels = np.concatenate((labels, subject_label), axis=0)

    assert sum(label_info['frames']) == labels.shape[0], 'process labels error...'
    return labels, label_info


def find_subjects(images_folders, subjects):
    our_images_folders = []

    for subject in subjects:
        for images_folder in images_folders:
            if len(images_folder.split('.')) == 1:
                if subject in images_folder:
                    our_images_folders.append(images_folder)

    return our_images_folders


def process_images(video_dir, label_info):

    images_folders = sorted(os.listdir(video_dir))
    images_folders = find_subjects(images_folders, label_info['subjects'])
    images_list = []
    success_list = []
    
    for images_folder, frames in zip(images_folders, label_info['frames']):
        images = sorted(os.listdir(os.path.join(video_dir, images_folder)))
        images_num = len(images)

        flag = 0
        if images_num > frames:
            print('image frames > labels')
            flag = -1
            images = images[:frames]
        if images_num < frames:
            print('image frames < labels')
            pad_num = frames - images_num
            flag = pad_num
            for i in range(pad_num):
                images += images[-1]

        for image in images:
            images_list.append(os.path.join(video_dir, images_folder, image))

        with open(os.path.join(video_dir, images_folder[:-8] + '.csv')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)
            success=[int(row[4]) for row in csv_reader]
            if flag == 0:
                success_list += success
            elif flag == -1:
                success_list += success[:frames]
            else:
                success_list += success
                for i in range(flag):
                    success_list += success[-1]

    assert len(images_list) == len(success_list), 'process images error ...'
    assert len(images_list) == sum(label_info['frames']), 'process images error ...'

    return images_list, success_list


def save_npy(save_dir, dataset_name, image_size, labels, images, success):
    labels_path = os.path.join(save_dir, dataset_name + '_label.npy')
    images_path = os.path.join(save_dir, dataset_name + '_images.npy')
    success_path = os.path.join(save_dir, dataset_name + '_success.npy')

    frames = labels.shape[0]
    fp_data = open_memmap(images_path, dtype='uint8', mode='w+', shape=(frames, image_size, image_size, 3))
    fp_label = open_memmap(labels_path, dtype='float32', mode='w+', shape=(frames, 12))
    fp_success = open_memmap(success_path, dtype='int', mode='w+', shape=(frames,))

    mean = [0.0] * 3
    std = [0.0] * 3
    valid_image_number = 0

    for i in tqdm(range(frames)):
        img = Image.open(images[i])
        img = np.array(img).astype(np.uint8)
        fp_data[i] = img
        fp_label[i] = labels[i]
        fp_success[i] = success[i]

        if success[i]:
            valid_image_number += 1

            r_mean = np.mean(img[:,:, 0]) / 255.0
            g_mean = np.mean(img[:,:, 1]) / 255.0
            b_mean = np.mean(img[:,:, 2]) / 255.0

            r_std = np.std(img[:,:, 0]) / 255.0
            g_std = np.std(img[:,:, 1]) / 255.0
            b_std = np.std(img[:,:, 2]) / 255.0

            mean[0] += r_mean
            mean[1] += g_mean
            mean[2] += b_mean

            std[0] += r_std
            std[1] += g_std
            std[2] += b_std

    mean = [value / valid_image_number for value in mean]
    std = [value / valid_image_number for value in std]

    return mean, std



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Disfa')
    parser.add_argument('--video_dir', type=str, default='./Videos_112_0.55')
    parser.add_argument('--label_dir', type=str, default='./Labels')
    parser.add_argument('--save_dir',  type=str, default='Dataset')
    parser.add_argument('--dataset_name',  type=str, default='Disfa_Right_112_0.55')
    parser.add_argument('--image_size', type=int, default=112)
    opts = parser.parse_args()

    if os.path.isdir(opts.save_dir) is False:
        os.mkdir(opts.save_dir)

    dataset_info = dict()
    
    labels, label_info = process_labels(opts.label_dir)
    images, success = process_images(opts.video_dir, label_info)

    mean, std = save_npy(opts.save_dir, opts.dataset_name, opts.image_size, labels, images, success)
    
    dataset_info['image_path']   = opts.dataset_name + '_images.npy'
    dataset_info['label_path']   = opts.dataset_name + '_label.npy'
    dataset_info['success_path'] = opts.dataset_name + '_success.npy'    
    dataset_info['label_info']   = label_info
    dataset_info['mean'] = mean
    dataset_info['std'] = std

    out_path = os.path.join(opts.save_dir, opts.dataset_name + '.json')
    with open(out_path, 'w') as json_file:
        json.dump(dataset_info, json_file)

    print('finish ...')