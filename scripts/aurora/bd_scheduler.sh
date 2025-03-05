#!/bin/bash 
#PBS -l select=100
#PBS -l walltime=06:00:00
#PBS -q prod
#PBS -l filesystems=home:flare
#PBS -A AuroraGPT
#PBS -M sakarvadia@uchicago.edu
#PBS -N softmax
#PBS -r y 

cd /lus/flare/projects/AuroraGPT/mansisak/distributed_ml/
module load frameworks
source env/bin/activate

cd /lus/flare/projects/AuroraGPT/mansisak/distributed_ml/src/experiments

pwd

python aurora_bd_scheduler.py --rounds 40 --checkpoint_every 5
