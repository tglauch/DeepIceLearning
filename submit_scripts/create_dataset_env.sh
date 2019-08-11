#!/bin/bash

echo $HOSTNAME
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
SDIR="$(dirname "$DIR")"
eval `/cvmfs/icecube.opensciencegrid.org/py2-v3.1.0/setup.sh`
source /data/user/tglauch/Software/combo_v2/build/env-shell.sh python "$SDIR/create_data_files.py" $@

