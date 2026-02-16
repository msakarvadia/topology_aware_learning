#!/bin/bash 
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -l filesystems=home:flare
#PBS -A AuroraGPT
#PBS -M sakarvadia@uchicago.edu
#PBS -N multi_node_ood
#PBS -r y 

cd /lus/flare/projects/AuroraGPT/mansisak/distributed_ml/
module load frameworks
source env/bin/activate

cd /lus/flare/projects/AuroraGPT/mansisak/distributed_ml/src/experiments

pwd

python extra_centrality_multi_node_ood.py --rounds 40 
