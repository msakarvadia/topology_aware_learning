#!/bin/bash 

JOBID="$(qsub label_hetero.sh 2>&1)"
echo "$JOBID";

max=4
for ((i = 0 ; i < max ; i++ )); 
do 
    JOBID="$(qsub -W depend=afterany:$JOBID label_hetero.sh 2>&1)"
    echo "$JOBID";
done
