#!/bin/bash

echo $HOSTNAME
eval `/cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/setup.sh`
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
SDIR="$(dirname "$DIR")"

source /cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/RHEL_7_x86_64/metaprojects/combo/stable/env-shell.sh python "$SDIR/create_data_files.py" $@

