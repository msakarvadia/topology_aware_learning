#!/bin/bash 
#PBS -l select=2
#PBS -l walltime=00:20:00
#PBS -q prod
#PBS -l filesystems=home:flare
#PBS -A AuroraGPT
#PBS -M sakarvadia@uchicago.edu
#PBS -N hetero_label
#PBS -r y 

cd /lus/flare/projects/AuroraGPT/mansisak/distributed_ml/
module load frameworks
source env/bin/activate

export TMPDIR=/tmp

cd /lus/flare/projects/AuroraGPT/mansisak/distributed_ml/src/experiments

pwd

python hetero_label.py --rounds 40 
