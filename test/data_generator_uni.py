import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from Gamma import auto_gamma


def normalize(a):
    return (a - np.min(a))/np.ptp(a)

def normalize_channels(a):
    for i in range(3):
        a[:,:,i] = normalize(a[:,:,i])
    return a

def gamma_correction(img, gamma):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(img, lookUpTable)

def normalize_meanstd(a, axis=None): 
    # axis param denotes axes along which mean & std reductions are to be performed
    a = a/255.
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / (std+1e-8)

def normalize_custom(a, mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]): 
    # axis param denotes axes along which mean & std reductions are to be performed
    a = a/255.
    for i in range(3):
        a[:,:,i] = (a[:,:,i] - mean[i]) / std[i]
    return a


class DataGeneratorW(Sequence):
    """
    Generates data for Keras
    ref: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(self,
                 X, Y, 
                 target_img_size,
                 batch_size,
                 gamma_adjust = True,
                 augment = ['rotate', 'blur', 'brightness'],
                 shuffle=True, return_weights = True):
        self.image_loc = X
        self.labels = Y
        self.data = np.column_stack([X, Y])
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data))
        self.target_img_size = target_img_size
        self.gamma_adjust = gamma_adjust
        self.augment = augment
        self.return_weights = return_weights
        self.on_epoch_end()

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
        
        X, labels_, samplew_ = self.__data_generation(batch_data)
        
        if self.return_weights:
            return X, labels_, samplew_
        else:
            return X, labels_
       
            

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_data):
        """
        Generates data containing batch_size samples
        :param annotation_lines:
        :return:
        """

        X = np.empty((len(batch_data), *self.target_img_size,3), dtype=np.float32)
        labels_ = np.empty((len(batch_data), len(batch_data[0])-2), dtype=np.float32)
        samples_ = np.empty((len(batch_data)), dtype=np.float32)
        for i, sample in enumerate(batch_data):
            img_data, label_,sample_ = self.get_data(sample)
            X[i] = img_data
            labels_[i] = label_
            samples_[i] = sample_
        return X, labels_, samples_
        

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
            
        #apply rotation if true:
        rotation = np.random.choice([cv2.ROTATE_90_CLOCKWISE,cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180])
        to_rotate = np.random.choice([True, False])
        
        to_flip = np.random.choice([True, False])
        
        
        augmentation = np.random.uniform(0,2.5, 1)[0]
        kernelsize = np.random.choice([3,5])
        to_augment = np.random.choice([True, False])
        
        brightness = int(np.random.uniform(0,30,1)[0])
        # to_brighten = np.random.choice([True, False])
        
        contrast = int(np.random.uniform(0.85,1.15,1)[0])
        to_contrast = np.random.choice([True, False])
        
        if self.gamma_adjust:
            train_image = auto_gamma(train_image)

        if 'blur' in self.augment:
            if to_augment:
                train_image = cv2.GaussianBlur(train_image, (kernelsize,kernelsize), augmentation)
        
        if 'rotate' in self.augment:
            if to_rotate:
                train_image = cv2.rotate(train_image, rotation)
        
        if 'flip' in self.augment:
            if to_flip:
                train_image = cv2.flip(train_image, 1) #horizontal flip
        
        if 'brightness' in self.augment:
            
            # if random choice is 1 then its increase the brighness else if -1 then decrease the brighness
            train_image  = cv2.convertScaleAbs(train_image, alpha=1.0, beta=brightness*np.random.choice([1, -1])) 
                
                
                    
        if 'contrast' in self.augment:
            if to_contrast:
                train_image  = cv2.convertScaleAbs(train_image, alpha=contrast, beta=0)
                
        
        # train_image = train_image/255.
        # train_image = normalize_custom(train_image)
        
        return train_image,  sample[1:-1].astype(np.float32),sample[-1]
        
def compute_weights(lines, normlaize_values = False):
    all_labels = []
    for line in lines:
        line = line.split()
        annotes = []
        for annos in line[1:]:
            #only append the last element (i.e. the class label)
            class_label = int(annos.split(',')[-1])
            annotes.extend([class_label])
        all_labels.extend(annotes)
    all_labels = np.array(all_labels)
    
    print("Total number of labels = {}".format(len(all_labels)))
    classes, counts= np.unique(all_labels, return_counts=True)
    for i in range(len(classes)):
        print('Total count for class {} : {}'.format(classes[i],counts[i]))
    
    class_wise_weights = len(counts)/(counts/counts.sum())
    if normlaize_values:
        class_wise_weights = class_wise_weights/class_wise_weights.sum()
    weights = dict(zip(classes, class_wise_weights))
    print("computed weights :{}".format(weights))
    return weights  

# import pandas as pd 
# df = pd.read_csv('train_labelsWsamples.csv')
# values = df.values
# print(values)
# dgen = DataGeneratorW(values[:,0],values[:,1:],  (224,224), 8, gamma_adjust= False, shuffle = False)
# for images, labels, samplew in dgen:
#     # images, masks = dgen.get_data(lines_v)
#     print(images.shape, labels.shape,samplew.shape)
#     print(labels,samplew)

#     for i in range(len(images)):
#         cv2.imshow('image', normalize(images[i][:,:,::-1]))
#         cv2.waitKey(500)
