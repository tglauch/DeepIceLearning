#!/bin/bash

echo $HOSTNAME
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
SDIR="$(dirname "$DIR")"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/mnt/lfs3/user/tglauch/envs/default/lib/python2.7/site-packages/:$PYTHONPATH
source /data/user/tglauch/Software/combo/build/env-shell.sh python "$SDIR/hdf_change.py" $@

