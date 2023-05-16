#!/bin/bash
for i in {1..20}
do
   echo "Running KRR $i"
   python3 KRR.py "$i"
   echo "Completed KRR $i"
   echo "Running GPR $i"
   python3 GPR.py "$i"
   echo "Completed GPR $i"

done
