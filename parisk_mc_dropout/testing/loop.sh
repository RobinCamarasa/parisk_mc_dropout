#!/bin/sh

for id in `seq 12 35`
do
    python testing.py --experimentobject-experiment-id $id
done
