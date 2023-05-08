#!/bin/bash
for i in $(seq 1 20)
do
   echo "Running GPR $i"
   python3 GPR.py "$i"
   echo "Completed GPR $i"
done

