# DeepIceLearning
DeepIceLearning - A software package to use Deep Neural Networks for Reconstructions and Classifications in IceCube

This software package is designed to make Deep Learning applications in IceCube as easy as possible. It contains a set of scripts to generate training datasets from i3 files, train Deep Neural Networks and apply them to data.

The main functionalities are provided by essentially three scripts and their corresponding submit files in the `/submit_scripts` folder. Addtional configurations for the training and dataset defintions can be done using the config files in `/config`. Neural networks are defined using the Keras functional API (see examples in `/networks/`). In addition to the Keras model definition python dictionaries in the network file are used in order to define the output and input features as well as transformations applied on them.

# 1. create_dataset.py

This file converts a set of i3files into training data. Training data files consist of 3d input tensors in the format 10x10x60 for IceCube and 3x5x60 for Deep core. The values to be stored in the tensor are defined in a config file (see `/configs/create_dataset_default.cfg`), classical examples are time at which x% of the charge are collected, the overall number of hits, total collected charge, charge after x ns, .. Also a set of `reco_vals` can be defined, i.e. quantities that are later used as target in the training of the network or for analyzing the network's performance.

In order create a MC dataset run something like

`python create_dataset_dagman.py --dataset_config file_to_dataset_config_file`

or add the `--data` option if you are using real data

The corresponding submit file for HTCondor can be used similarly 

`python create_dataset_dagman.py --dataset_config config_file_for_the_dataset --files_per_job num_i3files_per_job`

# 2. neural_network.py

This is the main script for training a network. As command line arguments it needs a config file that sets the hyperparameters and filepath for the training and a network definition file. Most of the network definition is done using standart Keras. Additionall transformations to the input and output data, as well as loss function settings can be defined as well. Compare the examples in `./networks/`

An example to run the training is

`bash nn_env.sh --main_config path_to_config_file --input all --model path_to_model_file`

or when sending the job to a (multi-gpu) cluster. Run from within the `submit_scripts` folder

`python submit_training.py --main_config path_to_config_file --input all --model path_to_model_file`

# 3. apply.py

Once a network is trained one might want to apply it to a set of MC data. For this just run something like 

`bash apply_env.sh --folder path_to_trained_network_folder --test_data path_to_the_test_data`

If you want to apply it to 'real' event data use instead

`bash apply_env.sh --folder path_to_trained_network_folder --data data_file(s)`


