#!/bin/bash

echo $HOSTNAME
. /data/user/tglauch/DeepIceLearning/python_env/bin/activate
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
SDIR="$(dirname "$DIR")"
export HDF5_USE_FILE_LOCKING=FALSE
source /data/user/tglauch/Software/combo/build/env-shell.sh python "$SDIR/hdf_change.py" $@

