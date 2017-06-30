# coding: utf-8

import os
import sys
import numpy as np
os.environ['THEANO_FLAGS'] = "device=gpu"  
import theano
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, BatchNormalization, MaxPooling2D,Convolution3D
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


input_data = np.array(np.load('./charge.npy'))
output_data = np.load('./truevals.npy')

print 'Shape of Input Data: {}'.format(np.shape(input_data))
print 'Shape of Output Data: {}'.format(np.shape(output_data))

tvt_ratio=[3,2,2] ##ratio of test validation and test dataset
data_len = len(input_data)
test_end = int(float(tvt_ratio[0])/np.sum(tvt_ratio)*data_len)
valid_end = int(float(tvt_ratio[1])/np.sum(tvt_ratio)*data_len)+test_end
print 'Range of training dataset {}:{}'.format(0,test_end)
print 'Range of validation dataset {}:{}'.format(test_end+1,valid_end)
print 'Range of test dataset {}:{}'.format(valid_end+1,data_len)

# # split train, validation and test samples
folders=['train_hist']
for folder in folders:
    if not folder in os.listdir('.'):
        os.makedirs('./{}'.format(folder))
train = input_data[0:test_end]
valid = input_data[test_end+1:valid_end]
test  = input_data[valid_end+1:data_len-1]
train_out = np.concatenate(output_data[0:test_end,0:1])
valid_out = np.concatenate(output_data[test_end+1:valid_end, 0:1])
test_out = np.concatenate(output_data[valid_end+1:data_len-1, 0:1])


# # ----------------------------------------------------------
# # Define model
# # ----------------------------------------------------------


def add_block(model, nfilters, dropout=False, **kwargs):
    """ 
    Add basic convolution block: 
     - 3x3 Convolution with padding
     - Activation: ReLU
     - either MaxPooling to reduce resolution, or Dropout
     - BatchNormalization
    """
    model.add(Convolution3D(nfilters, 5, 5,5, **kwargs)) #border_mode='same', init="he_normal", 
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    if dropout:
        model.add(Dropout(dropout))
    else:
        model.add(MaxPooling2D((2, 2), border_mode='same'))

def base_model():
  model = Sequential()
  model.add(Convolution3D(10, (3,3,3) , input_shape=(21, 21,51,1))) #border_mode='same', init="he_normal", 
  model.add(BatchNormalization())
  model.add(Activation('relu'))

  ##possible things to implement

  #model.add(Dropout(dropout))
  #model.add(MaxPooling2D((2, 2), border_mode='same'))

  model.add(Flatten()) 
  model.add(Dense(64))
  model.add(Activation('relu'))

  model.add(Dense(1, kernel_initializer='normal'))
  print(model.summary())

  model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
  return model


# # ----------------------------------------------------------
# # Training
# # ----------------------------------------------------------

estimator = KerasRegressor(build_fn=base_model, nb_epoch=1, batch_size=5, verbose=1)
seed = 7
np.random.seed(seed)

estimator.fit(np.expand_dims(train, axis=4),train_out, 
              validation_data=(np.expand_dims(valid, axis=4), valid_out),
              callbacks=[keras.callbacks.CSVLogger('./train_hist/history.csv')],
              verbose=1)
model.save('./train_hist/model.h5')  # save trained network
