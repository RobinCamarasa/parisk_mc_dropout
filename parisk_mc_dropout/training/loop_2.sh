#!/bin/sh

for validation in 0
do
    for dropout in 0.2 0.4 0.6
    do
        python training.py --model-dropout $dropout --datasplitter-validation $validation --model-dropout-type gaussian
    done
done
