#!/bin/bash
for i in $(seq 12 20)
do
   echo "Running GPR $i"
   python3 GPR.py "$i"
   echo "Completed GPR $i"
done

