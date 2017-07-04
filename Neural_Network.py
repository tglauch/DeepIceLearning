#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
os.environ['THEANO_FLAGS'] = "device=gpu, floatX = float32"  
os.environ["PATH"] += os.pathsep + '/usr/local/cuda/bin/nvcc'
import theano
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, BatchNormalization, MaxPooling2D,Convolution3D,MaxPooling3D
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.utils.io_utils import HDF5Matrix
from keras import regularizers
import h5py
import datetime
import gc


print('Loading Data...')

data_file='training_data/numu_train_data.h5'

# for obj in gc.get_objects():   # Browse through ALL objects
#     if isinstance(obj, h5py.File):   # Just HDF5 files
#         try:
#             obj.close()
#         except:
#             pass # Was already closed

# # input_data = data['charge'] ### Scale to a reasonable input 
# # output_data = data['reco_vals']

# print('Shape of Input Data: {}'.format(np.shape(input_data)))
# print('Shape of Output Data: {}'.format(np.shape(output_data)))

tvt_ratio=[10,1,1] ##ratio of test validation and test dataset
data_len = 100000 #len(h5py.File(data_file)['charge'])
test_end = int(float(tvt_ratio[0])/np.sum(tvt_ratio)*data_len)
valid_end = int(float(tvt_ratio[1])/np.sum(tvt_ratio)*data_len)+test_end
print('Range of training dataset {}:{}'.format(0,test_end))
print('Range of validation dataset {}:{}'.format(test_end+1,valid_end))
print('Range of test dataset {}:{}'.format(valid_end+1,data_len))

# # split train, validation and test samples
folders=['./train_hist', './train_hist/{}'.format(datetime.date.today())]
for folder in folders:
    if not os.path.exists('{}'.format(folder)):
        os.makedirs('{}'.format(folder))

print('Prepare Training Data Input')
######### Remove as soon as possible #########
# test_end = 100
# valid_end = 200
# data_len = 300
train = HDF5Matrix(data_file, 'charge', start=0, end=test_end)
valid = HDF5Matrix(data_file, 'charge', start=test_end+1, end=valid_end)
test  = HDF5Matrix(data_file, 'charge', start=valid_end+1, end=data_len-1)

######### Use log10(Energy) in order to avoid output values to go over several orders of magnitude ##############
reco_vals = HDF5Matrix(data_file, 'reco_vals')
print('Prepare Training Data Output')
# train_out = HDF5Matrix(data_file, 'reco_vals', start=0, end=test_end)
# valid_out = HDF5Matrix(data_file, 'reco_vals', start=test_end+1, end=valid_end)
# test_out  = HDF5Matrix(data_file, 'reco_vals', start=valid_end+1, end=data_len-1)
train_out = np.log10(reco_vals.data[0:test_end,0:1])
valid_out = np.log10(reco_vals.data[test_end+1:valid_end,0:1])
test_out = np.log10(reco_vals.data[valid_end+1:data_len-1,0:1])

def base_model():
  model = Sequential()

  model.add(Convolution2D(8, (3,3) , padding="same", kernel_initializer="he_normal",input_shape=(21, 21,51)))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.3))

  model.add(Convolution2D(8, (3,3), padding="same", kernel_initializer="he_normal"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D((3, 3), padding='same'))

  model.add(Convolution2D(8, (3,3), padding="same", kernel_initializer="he_normal"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D((3, 3), padding='same'))

  model.add(Flatten()) 
  model.add(Dropout(0.4))
  model.add(BatchNormalization())
  model.add(Dense(32,kernel_initializer='normal', activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))

  # model.add(Flatten(input_shape=(21, 21,51))) 
  # model.add(BatchNormalization())
  # model.add(Dense(128,kernel_initializer='normal', activation='relu',kernel_regularizer=regularizers.l2(0.01)))
  # model.add(Dropout(0.4))
  # model.add(Dense(64,kernel_initializer='normal', activation='relu',kernel_regularizer=regularizers.l2(0.01)))
  # model.add(BatchNormalization())
  # model.add(Dense(16,kernel_initializer='normal', activation='relu',kernel_regularizer=regularizers.l2(0.01)))
  # model.add(Dense(1, kernel_initializer='normal'))
  print(model.summary())

  model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
  return model


# # ----------------------------------------------------------
# # Training
# # ----------------------------------------------------------

estimator = KerasRegressor(build_fn=base_model, verbose=1)


estimator.fit(train,train_out, 
              validation_data=(valid , valid_out),
              callbacks=[keras.callbacks.CSVLogger('./train_hist/{}/history_noCNN_complete.csv'.format(datetime.date.today()))], 
              epochs=8, batch_size=50, verbose=1, shuffle='batch') #np.expand_dims(train, axis=4)
print('Save the model')
estimator.model.save('./train_hist/{}/last_model_noCNN_complete.h5'.format(datetime.date.today()))  # save trained network
print('calculate results...')
res= estimator.predict(test, verbose=1)

np.save('./train_hist/{}/last_model_noCNN_complete.npy'.format(datetime.date.today()), [res, np.squeeze(test_out)])
