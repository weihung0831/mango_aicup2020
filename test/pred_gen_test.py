import numpy as np
import cv2
import pandas as pd
import os
from tensorflow.keras.utils import Sequence




class TestGenerator(Sequence):
    """
    Generates data for Keras
    ref: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(self,
                 test_data ,target_size,
                 batch_size):
        
        self.data = test_data # format image_filename, x,y,w,h
        self.batch_size = batch_size
        self.target_size = target_size
        self.indexes = np.arange(len(self.data))
        
        
        
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
        
        imgs = self.__data_generation(batch_data)
        
        return imgs
       
            

    def __data_generation(self, batch_data):
        """
        Generates data containing batch_size samples
        :param annotation_lines:
        :return:
        """

        X = np.empty((len(batch_data), *self.target_size ,3), dtype=np.float32)
        for i, sample in enumerate(batch_data):
            img_data = self.get_data(sample)
            X[i] = img_data
        return X
        

    def get_data(self, sample):
        # print(annotation_line)
        
        img_path = sample[0]
        img_path = os.path.join('Test', sample[0])
        print(img_path)
        box = sample[1:]
        print(box)
        x,y,w,h = box
        # print(image_path)    
        
        train_image = cv2.imread(img_path)[:,:,::-1]
        # cv2.imshow('ixmg', train_image)
        # cv2.waitKey(700)
        train_image = train_image[y:y+h, x:x+w]
        train_image = cv2.resize(train_image,self.target_size)
        # cv2.imshow('ixmg', train_image)

        return train_image

# import pandas as pd 
# from time import sleep
# df = pd.read_csv('Test_mangoXYWH.csv')

# test_data = df.values

# # gen = 

# for images in TestGenerator(test_data,(224,224), 8):
#     for i in range(len(images)):
#         print(images.max(), images.min())
#         cv2.imshow('image', (images[i][:,:,::-1].astype('uint8')))
#         cv2.waitKey(30)