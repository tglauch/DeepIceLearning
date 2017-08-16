# DeepIceLearning
DeepIceLearning - An Approach to use Deep Neural Networks for the Regression and Classification of IceCube quantities

Firstly: This version of the software is still in test-mode and might contain many bugs or inconsitencys. 
As input you can use any files that are produced with the Create_Data_Files.py script in the main directory. A set of muon neutrino events can be found in my user directory under /data/user/tglauch/ML_Reco/training_data/

How to train a neural network with this code?
The goal of this software is to provide an easy usable framework for the training of neural network on IceCube data. In order to start
the training, mainly two files have to be changed.

1. The config.cfg defines all the variables for the local environment, as well as for the training. Here you can also set the path to the training files.

2. In the folder 'Networks' a config file defining the NN structure has to be created. An example for the syntax can be found in the 
file 'test.cfg'

In order to run the NN's training you can either:

1. directly run the script Neural_Network.py 
or 
2. submit a job to the condor or slurm cluster by running condor_submit/train_NN_network.py
