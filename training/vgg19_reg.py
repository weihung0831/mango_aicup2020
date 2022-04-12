#######################
from data_generator_uni import DataGeneratorW
from cosine_anneal import CosineAnnealingScheduler
from Gamma import auto_gamma
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from tqdm import tqdm
from pred_gen_auto import PredGenerator
from pred_gen_test import TestGenerator
import random
from keras.regularizers import l2

from sklearn.utils import class_weight
list_classes = ["1", "2", "3", "4", "5"]
def get_sample_weights(Y):
    mlb = MultiLabelBinarizer()
    mlb.fit([["1", "2", "3", "4", "5"],["1", "2", "3", "4",], ["3", "4", "5"]])
    # print(mlb.transform([["1", "2", "3", "4", "5"],["1", "2", "3", "4",], ["3", "4", "5"]]))
    mlb.inverse_transform(Y)
    sample_weights = class_weight.compute_sample_weight('balanced', Y)
    return np.array(sample_weights)

class changeReg(tf.keras.callbacks.Callback):
    def __init__(self, full_learning = 20):
        super(changeReg, self).__init__()
        self.regularizer = tf.Variable(1., dtype = tf.float32)
        self.full_learning = full_learning 

    def on_epoch_begin(self, epoch, logs={}):
        a = tf.constant([(epoch*epoch/self.full_learning)-7], dtype = tf.float32)
        self.regularizer.assign(tf.nn.sigmoid(a).numpy()[-1] )
        print("Setting regularization to =", str(self.regularizer))

class KerasTrainer: 
    
    def __init__(self, name = 'vgg19_tune',
                 num_classes = 5,
                 target_img_size=224,
                 batch_size=128,
                 augment = ['rotate', 'blur', 'brightness'],
                 loss = ['binary_crossentropy'],
                 metrics = ['acc'],
                 pretrain_lr = 1e-5,
                 reg_lr = 1e-6,
                 opt = 'SGD',
                 validation = 0.34,
                 pretrain_epoch = 20,
                 epochs = 50,):
        self.name = name
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.target_img_size = target_img_size
        self.augment = augment
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.pretrain_lr = pretrain_lr
        self.reg_lr = reg_lr
        self.pretrain_epoch = pretrain_epoch
        self.epochs = epochs
        
        self.validation = validation
        self.opt = opt
        self.history = None
        
        
        # define callbacks
        self.callbacks = None
        model_name = "models/"+self.name+"Epoch_{epoch:03d}-{loss:.3f}.h5"
 
        modelcheckpoint = ModelCheckpoint(model_name,
                                            monitor='loss',
                                            mode='auto',
                                            verbose=1,
                                            save_best_only=False)
        
        # define models
        #load resnet
        inputs = layers.Input((224, 224, 3))
        process_ = preprocess_input(inputs)
        model = VGG19(weights="imagenet", include_top=False, input_tensor= inputs, input_shape=(224, 224, 3))
        # model = NASNetMobile(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        #lock the resnet to not train
        model.trainable = False
        
        #add new classifier layers
        mod1 = model(process_)
                    
        # flat1= layers.GlobalAveragePooling2D()(mod1)
        flat1= layers.GlobalMaxPooling2D()(mod1)
        features = layers.Dense(512,activation=tf.nn.relu)(flat1)
        class5 = layers.Dense(512,activation=tf.nn.relu)(features)
        output= layers.Dense(5,activation=tf.nn.sigmoid, kernel_regularizer=l2(0.001))(class5)
        
        self.model = Model(inputs=inputs, outputs=output)
        # print model summary
        self.model.summary()
        # compile models
        self.callbacks = [modelcheckpoint]
        if self.opt =='SGD':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.pretrain_lr, momentum=0.9, nesterov=True)
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.pretrain_lr)
        self.model.compile(self.optimizer, loss = self.loss,  metrics = self.metrics)
        
    def load_weights(self, weight_file):
        self.model.trainable=True
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
            for i in range(self.pretrain_epoch):
                print('Pre-Training Epoch :::: {}/{} :::: Training on random split data'.format(i+1, self.pretrain_epoch))
                
                
                train_generator = DataGeneratorW(X_train,y_train, (self.target_img_size,self.target_img_size), 
                                                        self.batch_size,False, self.augment, return_weights = False)
                valid_generator = DataGeneratorW(X_test,y_test, (self.target_img_size,self.target_img_size), 
                                                        self.batch_size,False, self.augment, return_weights = False)
                    
                self.model.fit(train_generator, steps_per_epoch=len(train_generator), 
                                                    validation_data = valid_generator, validation_steps= len(valid_generator),
                                                    epochs=i+1, callbacks=self.callbacks,
                                                    initial_epoch =i, verbose = 1)
                
            self.model.save('models/pretrain_{}.h5'.format(self.name))
            
            
            self.model.trainable =True
                
            model_name = "models/"+self.name+"Post_Epoch_{epoch:03d}-{loss:.3f}.h5"    
            modelcheckpoint = ModelCheckpoint(model_name,
                                            monitor='loss',
                                            mode='auto',
                                            verbose=1,
                                            save_best_only=False)
            
            reg = changeReg(self.epochs)
            def loss_fn(y_true, y_pred):
                return reg.regularizer*self.loss[0](y_true, y_pred)
            self.callbacks = [modelcheckpoint, reg]
            self.loss_fn_= loss_fn
            if self.opt =='SGD':
                self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.reg_lr, momentum=0.9, nesterov=True)
            else:
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.reg_lr)
            
            
            self.model.compile(self.optimizer, loss = loss_fn,  metrics = self.metrics)
            self.model.summary()
            print('Now Training the model with regularized loss')
            
            
            for i in range(self.epochs):
                print('Regularized Epoch :::: {}/{} :::: Training on random split data'.format(i+1, self.epochs))
                
                
                train_generator = DataGeneratorW(X_train,y_train, (self.target_img_size,self.target_img_size), 
                                                        int(self.batch_size),False, self.augment, return_weights = False)
                valid_generator = DataGeneratorW(X_test,y_test, (self.target_img_size,self.target_img_size), 
                                                        int(self.batch_size),False, self.augment, return_weights = False)
                    
                self.model.fit(train_generator, steps_per_epoch=len(train_generator), 
                                                    validation_data = valid_generator, validation_steps= len(valid_generator),
                                                    epochs=i+1, callbacks=self.callbacks,
                                                    initial_epoch =i, verbose = 1)
                                
        print(self.history)
        self.model.save('models/{}.h5'.format(self.name))
    
    def predict_proba(self, X):
        generator = PredGenerator(X, np.zeros(len(X)), (self.target_img_size,self.target_img_size), self.batch_size, return_label=False)
        return self.model.predict(generator, steps=len(generator), verbose = 1)
    
    def predict_test(self, X):
        generator = TestGenerator(X, (self.target_img_size,self.target_img_size), self.batch_size)
        return self.model.predict(generator, steps=len(generator), verbose = 1)

    def evaluate(self, X, Y):
        generator = PredGenerator(X, Y, (self.target_img_size,self.target_img_size), self.batch_size, return_label=False)
        return self.model.predict(generator, steps=len(generator), verbose = 1)

    
        
    
