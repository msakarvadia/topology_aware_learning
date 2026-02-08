#!/bin/bash 
for i in 16 32 64 128 256;
do 
    JOBID="$(qsub -l select=$i dummy_experiment_launch_w_intercept_daos.sh )"
    #JOBID="$(qsub -l select=$i dummy_experiment_launch_wo_intercept_daos.sh )"
    echo "$JOBID";
done
