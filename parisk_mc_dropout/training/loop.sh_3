#!/bin/sh

for trainingnumbers in 3 5 9 15 20 40
do
    python training.py --model-dropout-type classic --model-dropout 0.1 --datasplitter-validation 3 --datasplitter-nb-training $training $trainingnumbers
done

