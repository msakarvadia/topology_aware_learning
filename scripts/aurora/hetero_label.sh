#!/bin/bash 
#PBS -l select=32
#PBS -l walltime=06:00:00
#PBS -q prod
#PBS -l filesystems=home:flare
#PBS -A AuroraGPT
#PBS -M sakarvadia@uchicago.edu
#PBS -N hetero_label
#PBS -r y 

cd /lus/flare/projects/AuroraGPT/mansisak/distributed_ml/
module load frameworks
source env/bin/activate

cd /lus/flare/projects/AuroraGPT/mansisak/distributed_ml/src/experiments

pwd

python hetero_label.py --rounds 40 
