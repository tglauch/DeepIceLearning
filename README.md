# DeepIceLearning
DeepIceLearning - An Approach to use Deep Neural Networks for the Regressions and Classifications in IceCube

This software package is designed to make Deep Learning applications in IceCube as easy as possible. It contains a set of scripts to generate training datasets from i3 files, train Deep Neural Networks and apply them to data.

The main functionalities are provided by essentially three scripts and their corresponding submit files in the `/submit_scripts` folder

# 1. create_dataset.py

This file converts a set of i3files into training data. Training data files consist of 3d input tensors in the format 11*10*60 for IceCube and 3*5*60 for Deep core. The values to be saved to the tensor are defined in a config file (see `/configs/create_dataset_default.cfg`), classical examples are time at which x% of the charge are collected, the overall number of hits, total collected charge...Also a set of 'reco_vals' can be defined which stores quantities that are later used as training output of the network or can be used for network analysis.

In order create a dataset run something like

`adsfajlsdaf`

# 2. neural_network.py

This is the main script for training a network. As command line arguments it needs a config file that sets the hyperparameters and filepath for the training and a network definition file. Most of the network definition is done using standart Keras. Additionall transformations to the input and output data, as well as loss function settings can be defined as well. Compare the examples in `./networks/`

An example to run the training on a GPU is

`bash nn_env.sh --main_config /scratch9/tglauch/DeepIceLearning/configs/main_tg.cfg --input all --model /scratch9/tglauch/DeepIceLearning/networks/new_multi.py`



