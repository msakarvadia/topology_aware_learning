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

for prox_coeff in 0;
do
    for aggregation_strategy in weighted;
    do
        for sample_alpha in 10;
        do
            for label_alpha in 10 100;
            do
                for topo in 8 7 6 5 4 3 2 1;
                do
                    topo_path=topology/topo_$topo.txt
                    python decentralized_main.py --topology $topo_path --aggregation_strategy $aggregation_strategy --sample_alpha $sample_alpha --label_alpha $label_alpha --prox_coeff $prox_coeff --rounds 5 --out_dir serial_logs
                done
            done
        done
    done
done
