#!/bin/bash

echo $PATH
echo $HOSTNAME
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
SDIR="$(dirname "$DIR")"
export KERAS_BACKEND="tensorflow"
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/compat/
export HDF5_USE_FILE_LOCKING='FALSE'
export TMPDIR=./tmp
export SINGULARITY_TMPDIR=./tmp
export SINGULARITY_CACHEDIR=./cache
singularity exec --nv -B /home/tglauch/:/home/tglauch/ -B /mnt/lfs3/user/:/data/user/ -B /mnt/lfs6/ana/:/data/ana/ -B /mnt/lfs6/sim/:/data/sim/ /data/user/tglauch/icetray_combo-stable-tensorflow.1.13.2-ubuntu18.04.sif /usr/local/icetray/env-shell.sh python "$SDIR/I3Module/i3module.py" $@
