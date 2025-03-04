#!/bin/bash 

JOBID="$(qsub daos_softmax.sh 2>&1)"
echo "$JOBID";

max=4
for ((i = 0 ; i < max ; i++ )); 
do 
    JOBID="$(qsub -W depend=afterany:$JOBID daos_softmax.sh 2>&1)"
    echo "$JOBID";
done
