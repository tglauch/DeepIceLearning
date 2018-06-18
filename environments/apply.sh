#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/

python Apply_Model.py --folder /scratch9/mkron/data/NN_out/$1 --main_config /scratch9/mkron/software/DeepIceLearning/configs/main_mk.cfg --batch_size 300 --model $2
