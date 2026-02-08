#!/bin/bash 
#PBS -l walltime=00:60:00
#PBS -q prod
#PBS -l filesystems=home:flare
#PBS -A AuroraGPT
#PBS -M sakarvadia@uchicago.edu
#PBS -N flare
#PBS -r y 

cd /lus/flare/projects/AuroraGPT/mansisak/distributed_ml/
module load frameworks
source env/bin/activate

cd /lus/flare/projects/AuroraGPT/mansisak/distributed_ml/src/experiments

pwd

num_nodes=$(wc -l < "${PBS_NODEFILE}")
exp_dir="flare/${num_nodes}_nodes"
echo "experiment dir: $exp_dir"
python dummy_experiments.py  --experiment_dir $exp_dir
