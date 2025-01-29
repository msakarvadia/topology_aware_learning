#!/bin/bash 
#PBS -l select=2
#PBS -l walltime=72:00:00
#PBS -q preemptable
#PBS -l filesystems=home:eagle
#PBS -A superbert
#PBS -M sakarvadia@uchicago.edu
#PBS -N softmax
#PBS -r y 

cd /eagle/projects/argonne_tpc/mansisak/distributed_ml 
module use /soft/modulefiles
module load conda
conda activate env/

cd /eagle/projects/argonne_tpc/mansisak/distributed_ml/src/experiments

NODES=`cat $PBS_NODEFILE | wc -l`
echo '# of nodes =' $NODES

python softmax.py --rounds 200 --checkpoint_every 10 --num_nodes $NODES
