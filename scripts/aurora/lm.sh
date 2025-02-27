#!/bin/bash 
#PBS -l select=31
#PBS -l walltime=01:00:00
#PBS -q debug-scaling
#PBS -l filesystems=home:flare
#PBS -A AuroraGPT
#PBS -M sakarvadia@uchicago.edu
#PBS -N lm
#PBS -r y 

cd /lus/flare/projects/AuroraGPT/mansisak/distributed_ml/
module load frameworks
source env/bin/activate

cd /lus/flare/projects/AuroraGPT/mansisak/distributed_ml/src/experiments

pwd

python lm.py --rounds 25 --checkpoint_every 1 --parsl_executor aurora_local
