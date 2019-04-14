#!/bin/bash
set -eux
GPUS=$(nvidia-smi --list-gpus | wc -l)
PYFILE=${1}
horovodrun -np $GPUS python3 $PYFILE train --cv-index=0
horovodrun -np $GPUS python3 $PYFILE train --cv-index=1
horovodrun -np $GPUS python3 $PYFILE train --cv-index=2
horovodrun -np $GPUS python3 $PYFILE train --cv-index=3
horovodrun -np $GPUS python3 $PYFILE train --cv-index=4
#python3 $PYFILE validate
#python3 $PYFILE predict
