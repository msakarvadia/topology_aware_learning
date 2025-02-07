#!/bin/bash 
#PBS -l select=2
#PBS -l walltime=72:00:00
#PBS -q preemptable
#PBS -l filesystems=home:eagle
#PBS -A superbert
#PBS -M sakarvadia@uchicago.edu
#PBS -N cifar10
#PBS -r y 

cd /eagle/projects/argonne_tpc/mansisak/distributed_ml 
module use /soft/modulefiles
module load conda
conda activate env/

cd /eagle/projects/argonne_tpc/mansisak/distributed_ml/src/experiments

NODES=`cat $PBS_NODEFILE | wc -l`
echo '# of nodes =' $NODES

python cifar10.py --rounds 50 --checkpoint_every 5 --num_nodes $NODES
