import numpy as np
import cv2
import pandas as pd
import operator
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from Gamma import auto_gamma
# from tensorflow.keras.applications.resnet_v2 import preprocess_input




class PredGenerator(Sequence):
    """
    Generates data for Keras
    ref: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(self,
                 X,Y ,
                 target_img_size,
                 batch_size, return_label = True):
        self.image_loc = X
        self.labels = Y
        self.data = np.column_stack([X, Y])
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.data))
        self.target_img_size = target_img_size
        
        self.return_label = return_label
        
        
    def __len__(self):
        'number of batches per epoch'
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch
        idxs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch_data = [self.data[i] for i in idxs]

        # Generate data
        
        X, labels_ = self.__data_generation(batch_data)
        # X = preprocess_input(X)
        if self.return_label:
            return X, labels_
        else:
            return X
       
            

    def __data_generation(self, batch_data):
        """
        Generates data containing batch_size samples
        :param annotation_lines:
        :return:
        """

        X = np.empty((len(batch_data), *self.target_img_size,3), dtype=np.float32)
        labels_ = np.empty((len(batch_data), len(batch_data[0])-1), dtype=np.float32)
        for i, sample in enumerate(batch_data):
            img_data, label_, = self.get_data(sample)
            X[i] = img_data
            labels_[i] = label_
            
        return X, labels_
        

    def get_data(self, sample):
        # print(annotation_line)
        
        img_path = sample[0]
        
        # print(image_path)
        resize_now = False
        if self.target_img_size[0] < 256:
            resize_now = True
            
        train_image = cv2.imread(img_path)[:,:,::-1]
        if resize_now:
            train_image = cv2.resize(train_image,self.target_img_size)

        return train_image,  sample[1:].astype(np.float32)
