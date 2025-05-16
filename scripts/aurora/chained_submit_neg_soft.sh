#!/bin/bash 

JOBID="$(qsub neg_soft.sh 2>&1)"
echo "$JOBID";

max=3
for ((i = 0 ; i < max ; i++ )); 
do 
    JOBID="$(qsub -W depend=afterany:$JOBID neg_soft.sh 2>&1)"
    echo "$JOBID";
done
