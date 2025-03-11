#!/bin/bash 

JOBID="$(qsub bd_scheduler.sh 2>&1)"
echo "$JOBID";

max=5
for ((i = 0 ; i < max ; i++ )); 
do 
    JOBID="$(qsub -W depend=afterany:$JOBID bd_scheduler.sh 2>&1)"
    echo "$JOBID";
done
