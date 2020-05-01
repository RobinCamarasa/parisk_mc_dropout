#!/bin/sh

for training in 0 1 2
do
    for dropout in 0.1 0.2 0.3 0.4
    do
        python training.py --model-dropout-type classic --model-dropout $dropout --datasplitter-validation 3 --datasplitter-training $training
        python training.py --model-dropout-type gaussian --model-dropout $dropout --datasplitter-validation 3 --datasplitter-training $training
    done
done

