
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
from sklearn.model_selection import RandomizedSearchCV
import itertools
import tables
import time


#read from npy file


# In[7]:

data =   np.load("datasets/updownclassification_using_keras_chargedata_filtered.npy")
labels = np.load("datasets/updownclassification_using_keras_labels_filtered.npy")

# In[9]:

#(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.25, random_state=42)
trainData, trainLabels = data, labels

#trying with onedimensional output layer
#(trainData, testData, trainLabels, testLabels) = train_test_split(data, np.array([0,1])[labels.argmax(1)], test_size=0.25, random_state=42)

# In[10]:

def create_model(neurons1=1024,neurons2=512, activations = ("relu", "relu")):
    model = Sequential()
    model.add(Dense(neurons1, input_dim=5160, kernel_initializer="uniform",
        activation=activations[0]))
    model.add(Dense(neurons2, kernel_initializer="uniform", activation=activations[1]))
    model.add(Dense(2))
    model.add(Activation("softmax"))
    #sgd = SGD(lr=0.01)
    adagrad = Adagrad()
    model.compile(loss="binary_crossentropy", optimizer=adagrad, metrics=["accuracy"])
    return model


#train the model
start=time.time()

timestamp = time.strftime("%d/%m %H:%M:%S")
orig_stdout = sys.stdout
f = open('logger.txt', 'a', buffering=0)
sys.stdout = f
print '\n'*2 + '----\n' + timestamp

model = KerasClassifier(build_fn=create_model, epochs = 20, batch_size = 128, verbose = 0)
print model.build_fn().summary()
"""
previously I just fitted with guessed params. now i do some fine tuning with gridsearch or randomizedsearch
history = model.fit(trainData, trainLabels, epochs=20, batch_size=128, verbose=1)

--- tuning optimizer ---
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer)


--- tuning layer sizes ---
firstlayersize=range(524,2025,100)
secondlayersize=range(112,1013,100)
param_grid = dict(neurons1=firstlayersize,neurons2=secondlayersize)
grid = RandomizedSearchCV(model, param_grid, n_iter=20, verbose=1)

--- tuning activation function



--- with gridsearch: 
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(trainData, trainLabels)
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

--- with randomizedsearch:
grid = RandomizedSearchCV(model, param_grid, n_iter=20, verbose=1)
grid_result = grid.fit(trainData, trainLabels)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
"""

act_fs = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
l1 = list(itertools.product(['relu'], act_fs))
l2 = list(itertools.product(act_fs, ['relu']))
param_grid = dict(activations = l1 + l2)
print "tuning with: ", param_grid
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)

# tune the hyperparameters via a randomized search

start = time.time()
grid_result = grid.fit(trainData, trainLabels)
# evaluate the best randomized searched model on the testing data
print("[INFO] randomized search took {:.2f} seconds".format(time.time() - start))

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
print "time to fit:",time.time()-start


# show the accuracy on the testing set
"""print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(testData, testLabels,
	batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))"""

#close file
sys.stdout = orig_stdout
f.close()


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
#bis jetzt mit vollem 000000-000999 first50 datensatz. jetzt mit gefiltert: |theta-90|>15
#parallel search anzahl neutrons tunen
"""
[Parallel(n_jobs=1)]: Done  60 out of  60 | elapsed: 139.3min finished
[Parallel(n_jobs=1)]: Done  60 out of  60 | elapsed: 169.1min finished
Fitting 3 folds for each of 20 candidates, totalling 60 fits
[INFO] randomized search took 8472.78 seconds
Fitting 3 folds for each of 20 candidates, totalling 60 fits
Best: 0.657148 using {'neurons1': 1824, 'neurons2': 612}
"""
