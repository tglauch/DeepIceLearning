
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
from keras.layers import Dropout
from keras.constraints import maxnorm
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
logfile = "logger.txt"

def train_model(data, labels, tuning=False, randomtune=True, n_iter=20, params=None, test_size = 0.25, logfile = None, verbose = 0, savefile=None, epochs = 20, batch_size = 128):
    if logfile:
        orig_stdout = sys.stdout
        f = open(logfile, 'a', buffering=0)
        sys.stdout = f
        timestamp = time.strftime("%d/%m %H:%M:%S")
        print '\n'*2 + '----\n' + timestamp
        
    
    if tuning:
        print "tuning model" + "with RandomSearchCV" if randomtune else "with GridSearchCV"
        trainData, trainLabels = data, labels
    else:
        print "Training model"
        #trying with onedimensional output layer
        #labels = np.array([0,1])[labels.argmax(1)]
        (trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=test_size, random_state=42)


    def create_model(neurons=(1824, 612), activations = ("relu", "softplus"), inits=("uniform","uniform"),\
                    dropout_rates=(0.0,0.0), weight_constraints=(0,0)):
        model = Sequential()
        model.add(Dense(neurons[0], input_dim=5160, kernel_initializer=inits[0],
                        activation=activations[0], 
                        kernel_constraint=None if dropout_rates[0] == 0.0 else maxnorm(weight_constraints[0])))
        model.add(Dropout(dropout_rates[0]))
        model.add(Dense(neurons[1], kernel_initializer=inits[1], activation=activations[1], 
                        kernel_constraint=None if dropout_rates[1] == 0.0 else maxnorm(weight_constraints[1])))
        model.add(Dropout(dropout_rates[1]))
        model.add(Dense(2))
        model.add(Activation("softmax"))
        #previously: sgd = SGD(lr=0.01)
        adagrad = Adagrad()
        if not tuning:
            print model.summary()
        model.compile(loss="binary_crossentropy", optimizer=adagrad, metrics=["accuracy"])
        return model

    start=time.time()
    
    
    if tuning:
        model = KerasClassifier(build_fn=create_model, epochs = epochs, batch_size = batch_size, verbose = verbose)
        print model.build_fn().summary()
        print "tuning with: ", params
        if randomtune: # tune the hyperparameters via a randomized search
            print "tuning randomly with RandomSearchCV"
            grid = RandomizedSearchCV(model, param_grid, n_iter=n_iter, verbose=1)
            grid_result = grid.fit(trainData, trainLabels)
        else:
            grid = GridSearchCV(estimator=model, param_grid=params, n_jobs=1)
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
   
    else:
        #train
        print "fitting..."
        #model = create_model()
        model = KerasClassifier(build_fn=create_model)
        model.fit(trainData, trainLabels, epochs = epochs, batch_size = batch_size, verbose=1, validation_data=(testData, testLabels))
        # show the accuracy on the testing set
        print("[INFO] evaluating on testing set...")
        (loss, accuracy) = model.evaluate(testData, testLabels,
            batch_size=128, verbose=1)
        print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
            accuracy * 100))
        #save model to file
        #if savefile:
            
        
    print "time to fit: {d}min {:.2f}sec".format(*(lambda t: (int(t/60),t%60))(time.time()-start))
    
    
    if logfile:
        #close file
        sys.stdout = orig_stdout
        f.close()
        
#def train_model(data, labels, tuning=False, randomtune=True, n_iter=20, params=None, test_size = 0.25, logfile = None, verbose = 0):    
        
#train_model(data, labels, epochs = 20)

#add a dropout between the two hidden layers and between the second hidden and the output layer
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param_grid = dict(dropout_rates=zip(dropout_rate,dropout_rate), weight_constraints=zip(weight_constraint,weight_constraint))
train_model(data, labels, tuning = True, randomtune=True, params=param_grid, logfile = logfile,n_iter=30)
    

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
act_fs = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
l1 = list(itertools.product(['relu'], act_fs))
l2 = list(itertools.product(act_fs, ['relu']))
param_grid = dict(activations = l1 + l2)

train_model(data, labels, tuning = True, params=param_grid, logfile = logfile)


---- tuning initializer:
init = ['uniform', 'normal']
param_grid = dict(inits = list(itertools.product(init,init)))

train_model(data, labels, tuning = True, randomtune=False, params=param_grid, logfile = None)
- Results: unifrom, uniform


--- tuning dropout rate
- training the network it could be seen that it overfits quickly. try to adda dropout layer



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
