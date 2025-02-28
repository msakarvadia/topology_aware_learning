#!/bin/bash 

JOBID="$(qsub lm.sh 2>&1)"
echo "$JOBID";

max=9
for ((i = 0 ; i < max ; i++ )); 
do 
    JOBID="$(qsub -W depend=afterany:$JOBID lm.sh 2>&1)"
    echo "$JOBID";
done
