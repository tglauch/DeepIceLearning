#!/bin/bash

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/
. /scratch9/tglauch/virtualenvs/dl_training/bin/activate
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
SDIR="$(dirname "$DIR")"
python "$SDIR/neural_network.py" $@
