#!/bin/bash

echo $HOSTNAME
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
SDIR="$(dirname "$DIR")"
export PYTHONPATH=$PYTHONPATH:/data/user/tglauch/envs/tf_env/lib/python2.7/site-packages
/cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/RHEL_7_x86_64/metaprojects/icerec/V05-02-03/env-shell.sh python "$SDIR/I3Module/i3module.py" $@


