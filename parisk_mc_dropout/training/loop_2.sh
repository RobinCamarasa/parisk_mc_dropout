#!/bin/sh

for validation in 0
do
    for dropout in 0.5 0.6
    do
        python training.py --model-dropout $dropout --datasplitter-validation $validation --model-dropout-type gaussian --trainer-nb-epochs 1000
    done
done
