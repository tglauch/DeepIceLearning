#!/bin/bash

#PY_ENV=/home/tglauch/virtualenvs/tf_env3/
IC_ENV=`/cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/setup.sh`
PY_ENV=/home/tglauch/venv_new/
echo $HOSTNAME
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
SDIR="$(dirname "$DIR")"
eval $IC_ENV
export HDF5_USE_FILE_LOCKING='FALSE'
export KERAS_BACKEND="tensorflow"
source /home/tglauch/i3/combo/build/env-shell.sh python "$SDIR/I3Module/i3module.py" $@


