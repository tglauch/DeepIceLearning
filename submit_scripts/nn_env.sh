#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
SDIR="$(dirname "$DIR")"
#source /cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/RHEL_7_x86_64/metaprojects/combo/stable/env-shell.sh python "$SDIR/neural_network.py" $@
python "$SDIR/neural_network.py" $@
#python /scratch9/mkron/software/DeepIceLearning/neural_network.py --main_config /scratch9/mkron/software/DeepIceLearning/configs/main_mk.cfg --input all --model /scratch9/mkron/software/DeepIceLearning/Networks/classifikation_mk/new_multi.py
