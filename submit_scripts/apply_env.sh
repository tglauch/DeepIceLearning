#!/bin/bash
echo $HOSTNAME
. /scratch9/tglauch/virtualenvs/dl_training/bin/activate
echo $LD_LIBRARY_PATH
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
SDIR="$(dirname "$DIR")"
python "$SDIR/apply.py" $@

