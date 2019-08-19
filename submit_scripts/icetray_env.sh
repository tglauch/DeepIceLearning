#!/bin/bash

echo $HOSTNAME
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
SDIR="$(dirname "$DIR")"
eval `/cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/setup.sh`
#echo $PYTHONPATH
#source /data/user/tglauch/envs/tf_env2/bin/activate
#export PYTHONPATH=/data/user/tglauch/envs/tf_env2/lib/python2.7/site-packages:$PYTHONPATH
export PYTONPATH=/home/tglauch/virtualenvs/tf_env3/lib/python2.7/site-packages:$PYTHONPATH
export HDF5_USE_FILE_LOCKING='FALSE'
#source /cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/RHEL_7_x86_64/metaprojects/simulation/V06-01-01/env-shell.sh python "$SDIR/run_icetray.py" $@
source /home/tglauch/i3/combo/build/env-shell.sh  python "$SDIR/run_icetray.py" $@
