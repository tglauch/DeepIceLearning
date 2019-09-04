#!/bin/bash

echo $HOSTNAME
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
SDIR="$(dirname "$DIR")"
eval `/cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/setup.sh`
export PYTONPATH=/home/tglauch/virtualenvs/tf_env3/lib/python2.7/site-packages:$PYTHONPATH
export HDF5_USE_FILE_LOCKING='FALSE'

source /home/tglauch/i3/combo/build/env-shell.sh python "$SDIR/I3Module/i3module.py" $@


