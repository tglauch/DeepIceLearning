#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/

python /scratch9/mkron/software/DeepIceLearning/neural_network.py --main_config /scratch9/mkron/software/DeepIceLearning/configs/main_mk.cfg --input all --model /scratch9/mkron/software/DeepIceLearning/networks/new_multi.py
