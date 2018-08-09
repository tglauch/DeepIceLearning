#!/bin/bash
echo $HOSTNAME
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
SDIR="$(dirname "$DIR")"
python "$SDIR/apply_model.py" $@

#python /scratch9/mkron/software/DeepIceLearning/apply_model.py --folder /scratch9/mkron/data/NN_out/$1 --main_config /scratch9/mkron/data/NN_out/$1/config.cfg --batch_size 30 --model $2 --weights $3
