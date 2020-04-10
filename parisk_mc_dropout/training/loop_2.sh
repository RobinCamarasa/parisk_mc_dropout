#!/bin/sh

for i in `seq 0 4`
do
    python /mnt/D2C8C54FC8C53291/parisk_projects/parisk_mc_dropout/parisk_mc_dropout/parisk_mc_dropout/training/training.py --model-dropout 0.3 --datasplitter-validation $i
done
