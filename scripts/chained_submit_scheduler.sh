#!/bin/bash 

JOBID="$(qsub scheduler.sh 2>&1)"
echo "$JOBID";

max=9
for ((i = 0 ; i < max ; i++ )); 
do 
    JOBID="$(qsub -W depend=afterany:$JOBID scheduler.sh 2>&1)"
    echo "$JOBID";
done
