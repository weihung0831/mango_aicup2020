#######################
from data_generator_uni import DataGeneratorW
from cosine_anneal import CosineAnnealingScheduler
from Gamma import auto_gamma
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from tqdm import tqdm
from pred_gen_auto import PredGenerator
import random

from sklearn.utils import class_weight
list_classes = ["1", "2", "3", "4", "5"]
def get_sample_weights(Y):
    mlb = MultiLabelBinarizer()
    mlb.fit([["1", "2", "3", "4", "5"],["1", "2", "3", "4",], ["3", "4", "5"]])
    # print(mlb.transform([["1", "2", "3", "4", "5"],["1", "2", "3", "4",], ["3", "4", "5"]]))
    mlb.inverse_transform(Y)
    sample_weights = class_weight.compute_sample_weight('balanced', Y)
    return np.array(sample_weights)


class KerasResnetTrainer: 
    
    def __init__(self, name = 'vgg16_tune',
                 num_classes = 5,
                 target_img_size=224,
                 batch_size=128,
                 as_SVM = False,
                 augment = ['rotate', 'blur', 'brightness'],
                 batch_norm = False,
                 balanced = True,
                 loss = ['binary_crossentropy'],
                 metrics = ['acc'],
                 cosine_annealing = True,
                 initial_lr = 1e-6,
                 opt = 'SGD',
                 validation = 0.34,
                 epochs = 100,):
        self.name = name
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.target_img_size = target_img_size
        self.augment = augment
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.batch_norm = batch_norm
        self.initial_lr = initial_lr
        self.epochs = epochs
        
        self.cosine_annealing = cosine_annealing
        self.balanced = balanced
        self.as_SVM = as_SVM
        self.validation = validation
        
        self.history = None
        
        
        # define callbacks
        self.callbacks = None
        model_name = "models/"+self.name+"Epoch_{epoch:03d}-{loss:.3f}.h5"
 
        modelcheckpoint = ModelCheckpoint(model_name,
                                            monitor='loss',
                                            mode='auto',
                                            verbose=1,
                                            save_best_only=True)
        
        if self.cosine_annealing:
            cosine = CosineAnnealingScheduler(self.epochs, 1e-7, self.initial_lr)
            self.callbacks = [modelcheckpoint, cosine]
        else:
            self.callbacks = [modelcheckpoint]
        
        # define models
        #load resnet
        inputs = layers.Input((224, 224, 3))
        process_ = preprocess_input(inputs)
        model = VGG16(weights="imagenet", include_top=False, input_tensor= inputs, input_shape=(224, 224, 3))
        # model = NASNetMobile(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        #lock the resnet to not train
        for layer in model.layers:
            if isinstance(layer, layers.BatchNormalization) and self.batch_norm:
                layer.trainable = True
            else:
                layer.trainable = False
        #add new classifier layers
        mod1 = model(process_)
        if as_SVM:
            flat1 = layers.GlobalAveragePooling2D()(mod1)
            class1 = layers.experimental.RandomFourierFeatures(2048*2, 
                                                                kernel_initializer='gaussian', 
                                                                scale=None, trainable=False)(flat1)
            output = layers.Dense(num_classes, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(class1)
        else:
                               
            # flat1 = layers.Flatten()(vae_conv)
            # flat1 = layers.GlobalMaxPooling2D()(model.layers[-1].output)
            # class1 = tfp.layers.DenseFlipout(256, kernel_divergence_fn=kl_divergence_function,activation=tf.nn.relu)(flat1)
            
            # output=tfp.layers.DenseFlipout(num_classes, kernel_divergence_fn=kl_divergence_function, activation = 'sigmoid')(class1)
            
            flat1 = layers.Flatten()(mod1)
            class1 = layers.Dense(512,activation=tf.nn.relu)(flat1)
            class1 = layers.Dense(128,activation=tf.nn.relu)(class1)
            
            output = layers.Dense(num_classes, activation = 'sigmoid')(class1)
            
        self.model = Model(inputs=inputs, outputs=output)
        # print model summary
        self.model.summary()
        # compile models
        
        if opt =='SGD':
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.initial_lr)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.initial_lr)
        self.model.compile(optimizer, loss = self.loss,  metrics = self.metrics)
        
    def load_weights(self, weight_file):
        self.model.load_weights(weight_file)
    
          
        
        
    def fit(self, X, y):
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.validation, random_state=43)
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=self.validation, random_state=0)
        for train_index, test_index in msss.split(X, y):
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
        print("TRAIN:", X_train.shape, y_train.shape, "TEST:", X_test.shape, y_test.shape)

        y_train = np.column_stack([y_train, get_sample_weights(y_train)])
        y_test = np.column_stack([y_test, get_sample_weights(y_test)])
        print('max sample weight: ',y_train[:,-1].max())
        # print(sample_weights)
        if self.validation:
            for i in range(self.epochs):
                print('Epoch :::: {}/{} :::: Training on random split data'.format(i+1, self.epochs))
                
                if self.balanced:
                    train_generator = DataGeneratorW(X_train,y_train, (self.target_img_size,self.target_img_size), 
                                                            self.batch_size, False, self.augment, return_weights = True)
                    
                    valid_generator = DataGeneratorW(X_test, y_test, (self.target_img_size,self.target_img_size), 
                                                            self.batch_size, False, self.augment, return_weights = False)
                
                    self.model.fit(train_generator, steps_per_epoch=len(train_generator), 
                                                    validation_data = valid_generator, validation_steps= len(valid_generator),
                                                    epochs=i+1, callbacks=self.callbacks,
                                                    initial_epoch =i,verbose = 1)
                else:
                    train_generator = DataGeneratorW(X_train,y_train, (self.target_img_size,self.target_img_size), 
                                                        self.batch_size,False, self.augment, return_weights = False)
                    valid_generator = DataGeneratorW(X_test,y_test, (self.target_img_size,self.target_img_size), 
                                                        self.batch_size,False, ['None'], return_weights = False)
                    
                    self.model.fit(train_generator, steps_per_epoch=len(train_generator), 
                                                    validation_data = valid_generator, validation_steps= len(valid_generator),
                                                    epochs=i+1, callbacks=self.callbacks,
                                                    initial_epoch =i, verbose = 1)
                
        else:
                        
            if self.balanced:
                train_generator = DataGeneratorW(X,y, (self.target_img_size,self.target_img_size), 
                                        self.batch_size, False, self.augment, return_weights = True)
                self.history = self.model.fit(train_generator, steps_per_epoch=len(train_generator),
                                epochs=self.epochs,
                                callbacks=self.callbacks,use_multiprocessing = False,
                                initial_epoch =i, verbose = 1)
            else:
                train_generator = DataGeneratorW(X,y, (self.target_img_size,self.target_img_size), 
                                        self.batch_size,False, self.augment, return_weights = False)
                self.history = self.model.fit(train_generator, steps_per_epoch=len(train_generator),
                                epochs=self.epochs,
                                callbacks=self.callbacks,use_multiprocessing = False,
                                initial_epoch =0, verbose = 1)
        print(self.history)
        self.model.save('models/single_estimator.h5')
  
    def predict(self, X, threshold = 0.5):
        for layer in self.model.layers:
            layer.trainable = False
        y_pred = self.predict_proba(X, batch = batch)
        return (y_pred > threshold) 
    
    def predict_proba(self, X):
        generator = PredGenerator(X, np.zeros(len(X)), (self.target_img_size,self.target_img_size), self.batch_size, return_label=False)
        return self.model.predict(generator, steps=len(generator), verbose = 1)

    def evaluate(self, X, Y):
        generator = PredGenerator(X, Y, (self.target_img_size,self.target_img_size), self.batch_size, return_label=False)
        return self.model.predict(generator, steps=len(generator), verbose = 1)
    
        
    
