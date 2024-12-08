#!/bin/bash 
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -l filesystems=home:eagle
#PBS -A superbert
#PBS -M sakarvadia@uchicago.edu

cd /eagle/projects/argonne_tpc/mansisak/distributed_ml 
module use /soft/modulefiles
module load conda
conda activate env/

cd /eagle/projects/argonne_tpc/mansisak/distributed_ml/src 

NODES=`cat $PBS_NODEFILE | wc -l`
echo '# of nodes =' $NODES

python sample_experiment.py --round 10 --checkpoint_every 10 --num_nodes $NODES
