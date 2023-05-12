#!/bin/bash
for i in 2 3 4 20
do
   echo "Running GPR $i"
   python3 GPR.py "$i"
   echo "Completed GPR $i"
done

