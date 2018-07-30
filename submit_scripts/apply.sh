#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/

python /scratch9/mkron/software/DeepIceLearning/Apply_Model.py --folder /scratch9/mkron/data/NN_out/$1 --main_config /scratch9/mkron/data/NN_out/$1/config.cfg --batch_size 30 --model $2 --weights $3
