#!/bin/sh

for validation in 0 1 2 3
do
    for trainingnumbers in 3 5 9 15 25 40
    do
        python training.py --model-dropout-type gaussian --model-dropout 0.2 --datasplitter-validation 3 --datasplitter-nb-training $training $trainingnumbers
        python training.py --model-dropout-type classic --model-dropout 0.2 --datasplitter-validation 3 --datasplitter-nb-training $training $trainingnumbers
    done
done

