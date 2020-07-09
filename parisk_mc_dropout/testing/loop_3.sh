#!/bin/sh

for id in `seq 168 181`
do
    python testing.py --experimentobject-experiment-id $id
done
