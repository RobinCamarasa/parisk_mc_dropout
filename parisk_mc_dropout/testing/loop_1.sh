#!/bin/sh

for id in `seq 48 59`
do
    python testing.py --experimentobject-experiment-id $id
done


for id in `seq 96 107`
do
    python testing.py --experimentobject-experiment-id $id
done
