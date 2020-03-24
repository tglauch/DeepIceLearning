#!/bin/bash

echo $HOSTNAME
echo $PATH
echo $PYTHONPATH
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
SDIR="$(dirname "$DIR")"
export KERAS_BACKEND="tensorflow"
export HDF5_USE_FILE_LOCKING=FALSE
if [ ! -e /usr/local/cuda/bin/ ]; then
    echo "Running on CPU!"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/compat/
else
   echo "Running on GPU!"
fi
export TMPDIR=/data/user/tglauch/tmp/
export SINGULARITY_TMPDIR=/data/user/tglauch/tmp/
export SINGULARITY_CACHEDIR=/data/user/tglauch/cache/
export DNN_BASE=/home/tglauch/I3Module/
export pythonpath=/usr/local/lib/
singularity exec --nv -B /scratch/condor/:/scratch/condor/ -B  /home/tglauch/:/home/tglauch/ -B /mnt/lfs7/user/:/data/user/ -B /mnt/lfs7/ana/:/data/ana/ -B /mnt/lfs6/sim/:/data/sim/ /data/user/tglauch/icetray_combo-stable-tensorflow.1.13.2-ubuntu18.04.sif /usr/local/icetray/env-shell.sh python $@
