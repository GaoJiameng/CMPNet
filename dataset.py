# -*- coding:utf-8 -*-
"""
@author: ryan
@software: PyCharm
@project name: CMPNet
@file: dataset.py
@time: 2021/09/08 21:15
@desc:
"""

import os
import cv2 
import random
import numpy as np
from PIL import Image
import paddle.vision.transforms as T
from tqdm import tqdm
from paddle.io import Dataset
from paddle.vision.transforms import Compose

from config import CONFIG, get

# View the distribution of the mean and standard deviation of the data set
def getStat(image_path_list):
    img_filenames = image_path_list
    m_list, s_list = [], []
    for img_filename in tqdm(img_filenames):
        img = cv2.imread(img_filename)
        img = img / 255.0
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True) * 256
    s = s_array.mean(axis=0, keepdims=True) * 256
    print(m[0][::-1])
    print(s[0][::-1])
    return m, s

def processing_dataset(wheat_images_path, home_path, train_list_Path, test_list_Path):
    # Guaranteed random reproducibility
    random.seed(8)

    # Get a list of wheat types
    wheat_type_list = get('LABEL_MAP')

    # Read all files and label them with the type of wheat
    data_list = []
    for i in wheat_type_list:
        temp_list = os.listdir(os.path.join(wheat_images_path, i))
        if '.DS_Store' in temp_list:
            temp_list.remove('.DS_Store')
        for j in temp_list:
            data_list.append([os.path.join(wheat_images_path, os.path.join(i, j)), str(wheat_type_list.index(i))])

    random.shuffle(data_list)

    # Separate training set, validation set and test set
    ratio = [0.8, 0.1]
    train_list = []
    eval_list = []
    test_list = []

    dataset_num = len(data_list)
    train_list = data_list[:int(dataset_num*ratio[0])]
    eval_list = data_list[int(dataset_num*ratio[0]):int(dataset_num*(ratio[0] + ratio[1]))]
    test_list = data_list[int(dataset_num*(ratio[0] + ratio[1])):]

    # Merging validation set and test set
    test_list.extend(eval_list)

    # Save training set and test set files
    temp_list = []
    new_train = []
    for i in train_list:
        new_train.append('{} {}\n'.format(i[0], i[1]))
    file = open(os.path.join(home_path, train_list_Path), "w")
    file.writelines(new_train)
    file.close()
    temp_list = []

    new_test = []
    for i in test_list:
        new_test.append('{} {}\n'.format(i[0], i[1]))
    file = open(os.path.join(home_path, test_list_Path), "w")
    file.writelines(new_test)
    file.close()
    temp_list = []

    # New total data set data_list
    data_list = []
    data_list_dir = []
    data_list.extend(train_list)
    data_list.extend(test_list)

    for i in range(len(data_list)):
        data_list_dir.append(data_list[i][0])

    # image_mean, image_std = getStat(data_list_dir)

    return train_list, test_list


def all_processing_dataset(Seedling_wheat_images_path, Flowering_wheat_images_path, Wheat_Seed_wheat_images_path, home_path, Seed_train_list_Path, Seed_test_list_Path, Flower_train_list_Path, Flower_test_list_Path, Wheat_Seed_train_list_Path, Wheat_Seed_test_list_Path):
    # Guaranteed random reproducibility
    random.seed(8)
    
    temp_seed = []
    temp_flower = []
    temp_wheat_seed = []
    difference_count = []
    train_list = []
    test_list = []

    Seed_train_list, Seed_test_list = processing_dataset(Seedling_wheat_images_path, home_path, Seed_train_list_Path, Seed_test_list_Path)
    Flower_train_list, Flower_test_list = processing_dataset(Flowering_wheat_images_path, home_path, Flower_train_list_Path, Flower_test_list_Path)
    Wheat_Seed_train_list, Wheat_Seed_test_list = processing_dataset(Wheat_Seed_wheat_images_path, home_path, Wheat_Seed_train_list_Path, Wheat_Seed_test_list_Path)

    # Merge data set
    Seed_train_list.extend(Seed_test_list)
    Flower_train_list.extend(Flower_test_list)
    Wheat_Seed_train_list.extend(Wheat_Seed_test_list)

    random.shuffle(Seed_train_list)
    random.shuffle(Flower_train_list)
    random.shuffle(Wheat_Seed_train_list)

    seed_nu = {"0":0, "1":0, "2":0, "3":0, "4":0, "5":0, "6":0, "7":0, "8":0, "9":0,
        "10":0, "11":0, "12":0, "13":0, "14":0, "15":0, "16":0, "17":0, "18":0, "19":0,
        "20":0, "21":0, "22":0, "23":0, "24":0, "25":0, "26":0, "27":0, "28":0, "29":0}
    for i in Seed_train_list:
        seed_nu[str(i[1])] = seed_nu[str(i[1])] + 1

    flower_nu = {"0":0, "1":0, "2":0, "3":0, "4":0, "5":0, "6":0, "7":0, "8":0, "9":0,
        "10":0, "11":0, "12":0, "13":0, "14":0, "15":0, "16":0, "17":0, "18":0, "19":0,
        "20":0, "21":0, "22":0, "23":0, "24":0, "25":0, "26":0, "27":0, "28":0, "29":0}
    for i in Flower_train_list:
        flower_nu[str(i[1])] = flower_nu[str(i[1])] + 1

    wheat_seed_nu = {"0":0, "1":0, "2":0, "3":0, "4":0, "5":0, "6":0, "7":0, "8":0, "9":0,
        "10":0, "11":0, "12":0, "13":0, "14":0, "15":0, "16":0, "17":0, "18":0, "19":0,
        "20":0, "21":0, "22":0, "23":0, "24":0, "25":0, "26":0, "27":0, "28":0, "29":0}
    for i in Wheat_Seed_train_list:
        wheat_seed_nu[str(i[1])] = wheat_seed_nu[str(i[1])] + 1
    
    wheat_label = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
          "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
          "20", "21", "22", "23", "24", "25", "26", "27", "28", "29"]

    # for i in wheat_label:
    #     difference_count.append(seed_nu[i] - flower_nu[i])

    for i in range(30):
        temp_seed = []
        temp_flower = []
        temp_wheat_seed = []

        for k in Seed_train_list:
            if int(k[1]) == int(i):
                temp_seed.append(k)
        for t in Flower_train_list:
            if int(t[1]) == int(i):
                temp_flower.append(t)
        for j in Wheat_Seed_train_list:
            if int(j[1]) == int(i):
                temp_wheat_seed.append(j)

        # Separate training set and test set
        ratio = [0.8, 0.2]
        temp_seed_train = []
        temp_seed_test = []
        temp_flower_train = []
        temp_flower_test = []
        temp_wheat_seed_train = []
        temp_wheat_seed_test = []

        dataset_num = len(temp_seed)
        temp_seed_train = temp_seed[:int(dataset_num*ratio[0])]
        temp_seed_test = temp_seed[int(dataset_num*ratio[0]):]
        dataset_num = len(temp_flower)
        temp_flower_train = temp_flower[:int(dataset_num*ratio[0])]
        temp_flower_test = temp_flower[int(dataset_num*ratio[0]):]
        dataset_num = len(temp_wheat_seed)
        temp_wheat_seed_train = temp_wheat_seed[:int(dataset_num*ratio[0])]
        temp_wheat_seed_test = temp_wheat_seed[int(dataset_num*ratio[0]):]

        for j in range(len(temp_seed_train)):
            train_list.append([temp_seed_train[j][0], temp_flower_train[j%len(temp_flower_train)][0], temp_wheat_seed_train[j%len(temp_wheat_seed_train)][0], str(temp_seed_train[j][1])])
        for j in range(len(temp_seed_test)):
            test_list.append([temp_seed_test[j][0], temp_flower_test[j%len(temp_flower_test)][0], temp_wheat_seed_test[j%len(temp_wheat_seed_test)][0], str(temp_seed_test[j][1])])

    random.shuffle(train_list)
    random.shuffle(test_list)

    # Save training set and test set files
    temp_list = []
    new_train = []
    for i in train_list:
        new_train.append('{} {} {} {}\n'.format(i[0], i[1], i[2], i[3]))
    file = open(os.path.join(home_path, "train_list.txt"), "w")
    file.writelines(new_train)
    file.close()
    temp_list = []

    new_test = []
    for i in test_list:
        new_test.append('{} {} {} {}\n'.format(i[0], i[1], i[2], i[3]))
    file = open(os.path.join(home_path, "test_list.txt"), "w")
    file.writelines(new_test)
    file.close()
    temp_list = []


class Reader(Dataset):
    def __init__(self, data, test=False):
        super().__init__()
        # In the initialization phase, the data set is divided into training set and test set. Since the samples have been shuffled before reading, take 20% of the samples as the test set and 80% of the samples as the training set
        self.samples = data
        self.test = test
        self.transform = Compose([
                T.RandomResizedCrop(224,scale=(0.8, 1),ratio=(1, 1)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], data_format='HWC'),
                T.ToTensor()
                ])

        self.transform1 = Compose([
                T.Resize(size=(224, 224)),
                T.Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], data_format='HWC'),
                T.ToTensor()
                ])
    def __getitem__(self, idx):

        seed_img_path = self.samples[idx][0]
        flower_img_path = self.samples[idx][1]
        wheat_seed_img_path = self.samples[idx][2]
        img_s = Image.open(seed_img_path)
        img_f = Image.open(flower_img_path)
        img_ws = Image.open(wheat_seed_img_path)
        if img_s.mode != 'RGB':
            img_s = img_s.convert('RGB')
        if img_f.mode != 'RGB':
            img_f = img_f.convert('RGB')
        if img_ws.mode != 'RGB':
            img_ws = img_ws.convert('RGB')

        if self.test == False:
            img_1 = self.transform(img_s)
            img_2 = self.transform(img_f)
            img_3 = self.transform(img_ws)
        else:
            img_1 = self.transform1(img_s)
            img_2 = self.transform1(img_f)
            img_3 = self.transform1(img_ws)


        label = self.samples[idx][3]
        label = np.array([label], dtype="int64")
        return img_1, img_2, img_3, label

    def __len__(self):
        """
        Get the total number of samples
        """
        return len(self.samples)





    





    













