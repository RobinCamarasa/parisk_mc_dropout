#!/bin/sh

python training.py --model-dropout-type classic --model-dropout 0.4 --datasplitter-validation 3 --datasplitter-nb-training 30
python training.py --model-dropout-type gaussian --model-dropout 0.4 --datasplitter-validation 3 --datasplitter-nb-training 30
python training.py --model-dropout-type classic --model-dropout 0.9 --datasplitter-validation 3 --datasplitter-nb-training 30
python training.py --model-dropout-type gaussian --model-dropout 0.9 --datasplitter-validation 3 --datasplitter-nb-training 30

