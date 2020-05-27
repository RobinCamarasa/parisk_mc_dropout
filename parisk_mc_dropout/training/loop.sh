#!/bin/sh

for trainingnumbers in 40 25 15 9 5 3
do
    python training.py --model-dropout-type variational --model-dropout 0.2 --datasplitter-validation 3 --datasplitter-nb-training $training $trainingnumbers
    python training.py --model-dropout-type variational --model-dropout 0.6 --datasplitter-validation 3 --datasplitter-nb-training $training $trainingnumbers
done

