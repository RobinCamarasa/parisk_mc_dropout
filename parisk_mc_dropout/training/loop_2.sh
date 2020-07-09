#!/bin/sh

python training.py --model-dropout-type classic --model-dropout 0.2 --datasplitter-validation 3 --datasplitter-nb-training 30
python training.py --model-dropout-type gaussian --model-dropout 0.2 --datasplitter-validation 3 --datasplitter-nb-training 30
python training.py --model-dropout-type classic --model-dropout 0.5 --datasplitter-validation 3 --datasplitter-nb-training 30
python training.py --model-dropout-type classic --model-dropout 0.7 --datasplitter-validation 3 --datasplitter-nb-training 30
python training.py --model-dropout-type gaussian --model-dropout 0.7 --datasplitter-validation 3 --datasplitter-nb-training 30

