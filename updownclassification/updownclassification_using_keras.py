
# coding: utf-8

# In[1]:

import os
import sys
import numpy as np
#os.environ['THEANO_FLAGS'] = "device=gpu, floatX = float32"  
#os.environ["PATH"] += os.pathsep + '/usr/local/cuda/bin/nvcc'
import theano
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.optimizers import Adagrad
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
import itertools
import tables
import time


#read from npy file


# In[7]:

data =   np.load("updownclassification_using_keras_chargedata.npy")
labels = np.load("updownclassification_using_keras_labels.npy")

# In[9]:

(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.25, random_state=42)

#trying with onedimensional output layer
#(trainData, testData, trainLabels, testLabels) = train_test_split(data, np.array([0,1])[labels.argmax(1)], test_size=0.25, random_state=42)

# In[10]:

def create_model(neurons=1024):
    model = Sequential()
    model.add(Dense(neurons, input_dim=5160, kernel_initializer="uniform",
        activation="relu"))
    model.add(Dense(512, kernel_initializer="uniform", activation="relu"))
    model.add(Dense(2))
    model.add(Activation("softmax"))
    #sgd = SGD(lr=0.01)
    adagrad = Adagrad()
    model.compile(loss="binary_crossentropy", optimizer=adagrad, metrics=["accuracy"])
    return model


# In[ ]:

#train the model
start=time.time()

model = KerasClassifier(build_fn=create_model, epochs = 20, batch_size = 128, verbose = 1)
#history = model.fit(trainData, trainLabels, epochs=20, batch_size=128, verbose=1)
#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#param_grid = dict(optimizer=optimizer)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)

# tune the hyperparameters via a randomized search
grid = RandomizedSearchCV(model, params)
start = time.time()
grid.fit(trainData, trainLabels)
# evaluate the best randomized searched model on the testing data
print("[INFO] randomized search took {:.2f} seconds".format(
	time.time() - start))

grid_result = grid.fit(trainData, trainLabels)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
print "time to fit:",time.time()-start

#plot history.history["acc"]

# In[80]:

# show the accuracy on the testing set
"""print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(testData, testLabels,
	batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))"""

# # results #
# feedforward neural net 5160-1024-512-2, relu activations. trained on 75% of 10501 datasets. tested on 25%.
# 
# Epoch 50/50
# 7875/7875 [==============================] - 3s - loss: 0.6672 - acc: 0.6000 
# 
# [INFO] evaluating on testing set...
# 2432/2626 [==========================>...] - ETA: 0s[INFO] loss=0.6787, accuracy: 56.5499%

# RESULTS GridSearch
"""
Best: 0.626794 using {'optimizer': 'Adagrad'}
0.549841 (0.011489) with: {'optimizer': 'SGD'}
0.575873 (0.012758) with: {'optimizer': 'RMSprop'}
0.626794 (0.008722) with: {'optimizer': 'Adagrad'}
0.568762 (0.015531) with: {'optimizer': 'Adadelta'}
0.597333 (0.008159) with: {'optimizer': 'Adam'}
0.599746 (0.001823) with: {'optimizer': 'Adamax'}
0.586413 (0.028847) with: {'optimizer': 'Nadam'}
"""

