#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/

python /scratch9/mkron/software/DeepIceLearning/Neural_Network.py --main_config /scratch9/mkron/software/DeepIceLearning/configs/main_mk.cfg --input all --model /scratch9/mkron/data/NN_out/run13/new_multi.py --continue /scratch9/mkron/data/NN_out/run13/ --load_weights /scratch9/mkron/data/NN_out/run13/best_val_loss.npy
