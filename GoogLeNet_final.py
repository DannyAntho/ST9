#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:15:11 2019

@author: dannyanthonimuthu
"""

print('fisk1')
import os
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, auc
#from keras.models import load_model
#from keras.callbacks import EarlyStopping
#import scipy.io as sio
from sklearn import metrics
import scipy.io as sio
#print(os.listdir("/Users/dannyanthonimuthu/Desktop/9_Sem/GoogLeNet_flowers/Cancer/"))


# define parameters
CLASS_NUM = 2
BATCH_SIZE = 256
EPOCH_STEPS = int(4323/BATCH_SIZE)
IMAGE_SHAPE = (256, 256, 3)
IMAGE_TRAIN = '/data'
#MODEL_NAME = 'googlenet_animals1.h5'
Validation_dir = '/val_data'
Test_dir = '/test_data'
output = '/output_data'

# prepare data
#train_datagen = ImageDataGenerator(
#    rescale=1./255,
    #rotation_range=30,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True
#)
    
train_datagen = ImageDataGenerator(rescale=1. / 255)    
vali_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(
                IMAGE_TRAIN,
                target_size=(256, 256),
                batch_size=BATCH_SIZE,
                class_mode='categorical',
                classes=['ITC+Mikro', 'No'],
                shuffle=True,)



validation_generator = vali_datagen.flow_from_directory(
                Validation_dir,
                target_size=(256, 256),
                batch_size=BATCH_SIZE,
                class_mode='categorical',
                classes=['ITC+Mikro', 'No'],
                shuffle=True,)

test_generator = vali_datagen.flow_from_directory(
                Test_dir,
                target_size=(256, 256),
                batch_size=BATCH_SIZE,
                class_mode='categorical',
                classes=['ITC_Mikro', 'No'],
                shuffle=True,)


y_trueVali = test_generator.classes

print('y_trueVali:',y_trueVali)
#print(train_generator)

#generator_main = train_datagen.flow_from_directory(
#    IMAGE_TRAIN,
#    target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
#    batch_size=BATCH_SIZE,
#    class_mode='categorical'
#)
'''
def my_generator(generator):
    while True: # keras requires all generators to be infinite
        data = next(generator)
        x = data[0]
        y = data[1]
        yield x, y
'''

#train_generator = my_generator(train_generator_one)

# create model
def inception(x, filters):
    # 1x1
    path1 = Conv2D(filters=filters[0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)

    # 1x1->3x3
    path2 = Conv2D(filters=filters[1][0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
    path2 = Conv2D(filters=filters[1][1], kernel_size=(3,3), strides=1, padding='same', activation='relu')(path2)
    
    # 1x1->5x5
    path3 = Conv2D(filters=filters[2][0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
    path3 = Conv2D(filters=filters[2][1], kernel_size=(5,5), strides=1, padding='same', activation='relu')(path3)

    # 3x3->1x1
    path4 = MaxPooling2D(pool_size=(3,3), strides=1, padding='same')(x)
    path4 = Conv2D(filters=filters[3], kernel_size=(1,1), strides=1, padding='same', activation='relu')(path4)

    return Concatenate(axis=-1)([path1,path2,path3,path4])


def auxiliary(x, name=None):
    layer = AveragePooling2D(pool_size=(5,5), strides=3, padding='valid')(x)
    layer = Conv2D(filters=128, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)
    layer = Flatten()(layer)
    layer = Dense(units=256, activation='relu')(layer)
    layer = Dropout(0.4)(layer)
    layer = Dense(units=CLASS_NUM, activation='softmax', name=name)(layer)
    return layer


def googlenet():
    layer_in = Input(shape=IMAGE_SHAPE)
    
    # stage-1
    layer = Conv2D(filters=64, kernel_size=(7,7), strides=2, padding='same', activation='relu')(layer_in)
    layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
    #layer = BatchNormalization()(layer)

    # stage-2
    layer = Conv2D(filters=64, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)
    layer = Conv2D(filters=192, kernel_size=(3,3), strides=1, padding='same', activation='relu')(layer)
    #layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)

    # stage-3
    layer = inception(layer, [ 64,  (96,128), (16,32), 32]) #3a
    layer = inception(layer, [128, (128,192), (32,96), 64]) #3b
    layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
    
    # stage-4
    layer = inception(layer, [192,  (96,208),  (16,48),  64]) #4a
    aux1  = auxiliary(layer, name='aux1')
    layer = inception(layer, [160, (112,224),  (24,64),  64]) #4b
    layer = inception(layer, [128, (128,256),  (24,64),  64]) #4c
    layer = inception(layer, [112, (144,288),  (32,64),  64]) #4d
    aux2  = auxiliary(layer, name='aux2')
    layer = inception(layer, [256, (160,320), (32,128), 128]) #4e
    layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
    
    # stage-5
    #layer = inception(layer, [256, (160,320), (32,128), 128]) #5a
    #layer = inception(layer, [384, (192,384), (48,128), 128]) #5b
    #layer = AveragePooling2D(pool_size=(7,7), strides=1, padding='valid')(layer)
    #layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
    
    # stage-6
    layer = Flatten()(layer)
    layer = Dropout(0.4)(layer)
    layer = Dense(units=256, activation='linear')(layer)
    main = Dense(units=CLASS_NUM, activation='softmax', name='main')(layer)
    
    model = Model(inputs=layer_in, outputs=[main]) #input og output 
    
    return model

# train model
model = googlenet()
model.summary()


#model.load_weights(MODEL_NAME)
#tf.keras.utils.plot_model(model, 'GoogLeNet.png')

optim = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.1)
#optimizer = SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

#optimizer = ['Adam', 'SGD', 'Adam', 'SGD']
#epochs = [20, 30, 20, 30]
#optimizer = ['Adam']
#epochs = [1]
#history_all = {}

#or i in range(len(optimizer)):
#    print('Using optimizer: ' + optimizer[i] + ', Epoch: ' + str(epochs[i]))

print('fisk2')    
model.compile(loss='categorical_crossentropy', 
                  optimizer=optim, metrics=['accuracy'])

print('fisk3')

CLASS_WEIGHTS = {0 : 15, 1 : 1}

    
train_history = model.fit_generator(
    train_generator,
    steps_per_epoch=EPOCH_STEPS,
    epochs=35,
    #callbacks= callbacks,
    validation_data = validation_generator,
    class_weight = CLASS_WEIGHTS,
    shuffle=True
    )
print('fisk4')
 
    
#model.evaluate_generator(train_generator,steps=train_generator.n/BATCH_SIZE)    

#model.save('GoogleAnimal1.hdf5')    


Prediction =model.predict_generator(test_generator, steps=validation_generator.n/BATCH_SIZE)
y_predVali = np.argmax(Prediction, axis = 1)

print('y_predVali', y_predVali)

''' Gem ting '''
sio.savemat(output + '/y_predVali.mat',{'y_predVali': y_predVali})
sio.savemat(output + '/y_trueVali.mat',{'y_trueVali': y_trueVali})


F1_all = metrics.classification_report(y_trueVali, y_predVali, digits=5)

print(F1_all)
  
Matrix = confusion_matrix(y_trueVali, y_predVali)
print('Confusion:',Matrix)   


F1_score = f1_score(y_trueVali, y_predVali, average="macro")

print('F1 Score:' ,F1_score)

precision = dict()
recall = dict()
#pr_auc = list()

for i in range(CLASS_NUM): 
  precision[i], recall[i], _ = precision_recall_curve(y_trueVali,Prediction[:,i], pos_label =i)
  #pr_auc[i] = auc(recall[i],precision[i])
  

print('precision', precision)  

print('recall', recall)



savPrec = output + '/precision.npy'
savRec = output + '/recall.npy'

np.save(savPrec, precision)
np.save(savRec,recall)





#lr_precision, lr_recall, thresholds = precision_recall_curve(y_trueVali, y_predVali)
#print('Precision:', lr_precision)
#print('Recall:', lr_recall)
#print('Thresholds', thresholds)

print('fisk500')





