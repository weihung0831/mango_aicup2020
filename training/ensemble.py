#######################
from data_generator_uni import DataGeneratorW
from cosine_anneal import CosineAnnealingScheduler
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50V2, NASNetMobile, DenseNet121, EfficientNetB5
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import random

class EnsembleTrainer: 
    
    def __init__(self, name = 'ensemble_model',
                 num_classes = 5,
                 target_img_size=224,
                 batch_size=128,
                 augment = ['rotate', 'blur', 'brightness'],
                 batch_norm = False,
                 loss = ['binary_crossentropy'],
                 metrics = ['acc'],
                 cosine_annealing = True,
                 initial_lr = 1e-6,
                 opt = 'SGD',
                 validation = None,
                 epochs = 100, balanced = True):
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
        #here is the input
        inputs = layers.Input((224, 224, 3))
        
        # process_res = tf.keras.applications.resnet_v2.preprocess_input(inputs)
        process_nas = tf.keras.applications.nasnet.preprocess_input(inputs)
        # process_dense = tf.keras.applications.nasnet.preprocess_input(inputs)
        process_eff = tf.keras.applications.efficientnet.preprocess_input(inputs)
        # model1 = ResNet50V2(weights="imagenet", include_top=False, input_tensor= inputs, input_shape=(224, 224, 3))
        # model1 = DenseNet121(weights="imagenet", include_top=False, input_tensor= inputs, input_shape=(224, 224, 3))
        model1 = EfficientNetB5(weights="imagenet", include_top=False, input_tensor= inputs, input_shape=(224, 224, 3))
        model2 = NASNetMobile(weights="imagenet", include_top=False, input_tensor= inputs, input_shape=(224, 224, 3))
        
        
        
        #lock the resnet to not train
        for layer in model1.layers:
            try:
                if isinstance(layer, layers.BatchNormalization) and self.batch_norm:
                    layer.trainable = True
                else:
                    layer.trainable = False
            except:
                print('its a tensor')
        
        for layer in model2.layers:
            if isinstance(layer, layers.BatchNormalization) and self.batch_norm:
                layer.trainable = True
            else:
                layer.trainable = False
        
        mod1 = model1(process_eff)
        mod2 = model2(process_nas)
        #add new classifier layers
        flat1 = layers.GlobalAveragePooling2D()(mod1)
        flat2 = layers.GlobalAveragePooling2D()(mod2)
        
        
        class1 = layers.Dense(1028, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-3))(flat1)
        class1 = layers.Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-3))(class1)
        
        class2 = layers.Dense(512, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-3))(flat2)
        class2 = layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-3))(class2)
        
        output1 = layers.Dense(num_classes, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(1e-3))(class1)
        output2 = layers.Dense(num_classes, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(1e-3))(class2)
        
        average = layers.average([output1, output2])
        self.model = Model(inputs=inputs, outputs=average)
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
    
    def build_new(self, ):
        # del self.model
        # tf.keras.backend.clear_session()
        a = None
        
        
        
    def fit(self, X, y):
        
        if self.validation:
            for i in range(self.epochs):
                print('Epoch : {} :::: Training on new random split'.format(i))
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.validation, random_state=int(random.random()*100))
                if self.balanced:
                    train_generator = DataGeneratorW(X_train,y_train, (self.target_img_size,self.target_img_size), 
                                                            self.batch_size, False, self.augment, return_weights = True)
                    
                    valid_generator = DataGeneratorW(X_test, y_test, (self.target_img_size,self.target_img_size), 
                                                            self.batch_size, False, ['None'], return_weights = False)
                
                    self.model.fit(train_generator, steps_per_epoch=len(train_generator), 
                                                    validation_data = valid_generator, validation_steps= len(valid_generator),
                                                    epochs=i+1, callbacks=self.callbacks,
                                                    initial_epoch =i, verbose = 1)
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
                                initial_epoch =0, verbose = 1)
            else:
                train_generator = DataGeneratorW(X,y, (self.target_img_size,self.target_img_size), 
                                        self.batch_size,False, self.augment, return_weights = False)
                self.history = self.model.fit(train_generator, steps_per_epoch=len(train_generator),
                                epochs=self.epochs,
                                callbacks=self.callbacks,use_multiprocessing = False,
                                initial_epoch =0, verbose = 1)
        print(self.history)
        self.model.save('models/ensemble.h5')
  
    def predict(self, X, threshold = 0.5):
        for layer in self.model.layers:
            layer.trainable = False
        y_pred = self.predict_proba(X)
        return (y_pred > threshold) 
    
    def predict_proba(self, X):
        probabilities = np.zeros((len(X),self.num_classes))
        for i in range(len(X)):
            img = cv2.imread(X[i])[:,:,::-1]
            img = cv2.resize(img, (self.target_img_size,self.target_img_size))
            img  = img.reshape((1,self.target_img_size,self.target_img_size,3))
            
            probabilities[i] = self.model.predict(img)
        return probabilities
    
        
    
