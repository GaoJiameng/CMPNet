# -*- coding:utf-8 -*-
"""
@author: ryan
@software: PyCharm
@project name: CMPNet
@file: config.py
@time: 2021/09/08 21:15
@desc: 
"""

# image dataset path
home_path = "./"
Seedling_wheat_images_path = "./Seedling_wheat_images"
Flowering_wheat_images_path = "./Flowering_wheat_images"
Wheat_Seed_wheat_images_path = "./Wheat_seed_data"

Seed_train_list_Path = "Seed_train_list.txt"
Seed_test_list_Path = "Seed_test_list.txt"
Flower_train_list_Path = "Flower_train_list.txt"
Flower_test_list_Path = "Flower_test_list.txt"
Wheat_Seed_train_list_Path = "Wheat_Seed_train_list.txt"
Wheat_Seed_test_list_Path = "Wheat_Seed_test_list.txt"

all_train_list_Path = "train_list.txt"
all_test_list_Path = "test_list.txt"
species_list = "species.txt"

CONFIG = {
    'boundaries' : [5,10,15],
    'PWDvalues' : [0.0005,0.0001,0.00002,0.00001],
    'L2' : 0.2,
    'topk' : (1, 2),
    'batch_size' : 64,
    'epochs' : 12,
    'save_dir' : "./lup",
    'save_freq' : 20,
    'log_freq' : 100,
    'LABEL_MAP': ['LT19', 'LT58', 'ZM19', 'ZM21', 'LT55', 'JM44',
                    'JM21', 'LT15', 'JM20', 'LT54', 'JM47', 'ZM22',
                    'LT53', 'JM19', 'LT43', 'LT36', 'LT26', 'LT37',
                    'LT45', 'LT34', 'LT33', 'LT56', 'ZM23', 'LT39',
                    'LT42', 'LT40', 'LT35', 'LT48', 'JM22', 'ZM20']
    }


def get(full_path):
    for id, name in enumerate(full_path.split('.')):
        if id == 0:
            config = CONFIG
        
        config = config[name]
    
    return config