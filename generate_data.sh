#!/bin/bash
for i in 1
do
   echo "Running KRR $i"
   python3 GPR.py "$i"
   echo "Completed KRR $i"
done

