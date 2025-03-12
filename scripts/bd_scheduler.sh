#!/bin/bash 
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -q preemptable
#PBS -l filesystems=home:eagle
#PBS -A AuroraGPT
#PBS -M sakarvadia@uchicago.edu
#PBS -N bd_schedule
#PBS -r y 

cd /eagle/projects/argonne_tpc/mansisak/distributed_ml 
module use /soft/modulefiles
module load conda
conda activate env/

cd /eagle/projects/argonne_tpc/mansisak/distributed_ml/src/experiments

python bd_scheduler.py --rounds 30 --checkpoint_every 5 
