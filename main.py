# -*- coding:utf-8 -*-
"""
@author: ryan
@software: PyCharm
@project name: CMPNet
@file: main.py
@time: 2021/09/08 21:15
@desc: 
"""

import os
import cv2
import glob
import paddle
import random
import pandas as pd
from paddle.nn import Linear
from paddle.nn import Conv2D, MaxPool2D
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from net import MixedNN
from config import *
from dataset import processing_dataset, all_processing_dataset, Reader


# set random seed
random.seed(8)

# Pack data into categories
all_processing_dataset(Seedling_wheat_images_path, Flowering_wheat_images_path, Wheat_Seed_wheat_images_path, home_path, Seed_train_list_Path, Seed_eval_list_Path, Flower_train_list_Path, Flower_eval_list_Path, Wheat_Seed_train_list_Path, Wheat_Seed_eval_list_Path)

train_list = []
test_list = []

# Create training data path array
with open(all_train_list_Path) as f:
    for line in f:
        a, b, c, d = line.strip("\n").split(" ")
        train_list.append([a, b, c, int(d)])

# Create test data path array
with open(all_test_list_Path) as f:
    for line in f:
        a, b, c, d = line.strip("\n").split(" ")
        test_list.append([a, b, c, int(d)])

# Generate training data set instance
train_dataset = Reader(train_list)

# Generate test data set instance
test_dataset = Reader(test_list, test=True)

# Declare the defined model
model = MixedNN()
# The model structure is expected to generate model objects to facilitate subsequent configuration, training and verification
model = paddle.Model(model)

learning_rate=paddle.optimizer.lr.PiecewiseDecay(boundaries=get('boundaries'),
                                    values=get('PWDvalues'),
                                    # verbose=True
                                    )

model.prepare(paddle.optimizer.Adam(
                                    learning_rate=learning_rate,
                                    parameters=model.parameters(),
                                    weight_decay=paddle.regularizer.L2Decay(get('L2'))
                                    ),
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy(topk=get('topk')))  # Configure Accuracy evaluation indicators

visualdl=paddle.callbacks.VisualDL(log_dir='visual_log')

model.fit(train_data=train_dataset,       # Training data set
          batch_size=get('batch_size'),   # The number of samples in a batch
          epochs=get('epochs'),           # Iteration round
          shuffle=True,                   # Each EPOCH scrambles the sample once, which improves the training effect a little
          verbose=1,
          save_dir=get('save_dir'),       # Save model parameters and optimizer parameters to a custom folder
          save_freq=get('save_freq'),     # Set how many epochs to save model parameters and optimizer parameters
          log_freq=get('log_freq')        # How often to print the log
          callbacks=[visualdl]
)

# load parameters
model.load('lup/final.pdparams')
# evaluate model
model.evaluate(test_dataset, batch_size=1000, verbose=1)