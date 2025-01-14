#!/bin/bash 
#PBS -l select=10
#PBS -l walltime=03:00:00
#PBS -q prod
#PBS -l filesystems=home:eagle
#PBS -A superbert
#PBS -M sakarvadia@uchicago.edu
#PBS -N hetero
#PBS -r y 

cd /eagle/projects/argonne_tpc/mansisak/distributed_ml 
module use /soft/modulefiles
module load conda
conda activate env/

cd /eagle/projects/argonne_tpc/mansisak/distributed_ml/src/experiments

NODES=`cat $PBS_NODEFILE | wc -l`
echo '# of nodes =' $NODES

python heterogeneous.py --rounds 300 --checkpoint_every 20 --num_nodes $NODES
