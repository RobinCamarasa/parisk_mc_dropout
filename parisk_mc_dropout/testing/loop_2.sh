#!/bin/sh

for id in `seq 118 143`
do
    python testing.py --experimentobject-experiment-id $id
done

