#!/bin/sh

for validation in 0 1 2 3
do
    for dropout in 0.1 0.3 0.5
    do
        python training.py --model-dropout $dropout --datasplitter-validation $validation --model-dropout-type gaussian
    done
done
