#!/bin/bash
set -eux
GPUS=$(nvidia-smi --list-gpus | wc -l)
PYFILE=$1
mpirun -np $GPUS python3 $PYFILE train --cv-index=0
mpirun -np $GPUS python3 $PYFILE train --cv-index=1
mpirun -np $GPUS python3 $PYFILE train --cv-index=2
mpirun -np $GPUS python3 $PYFILE train --cv-index=3
mpirun -np $GPUS python3 $PYFILE train --cv-index=4
python3 $PYFILE validate --cv
python3 $PYFILE predict --cv
