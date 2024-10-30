#!/bin/bash

cd /eagle/projects/argonne_tpc/mansisak/distributed_ml/src 

for prox_coeff in 0 0.1 0.5;
do
    for aggregation_strategy in weighted unweighted;
    do
        for sample_alpha in 100 1000;
        do
            for label_alpha in 100 1000;
            do
                for topo in 1 2 3 4 5 6 7;
                do
                    topo_path=topology/topo_$topo.txt
                    python decentralized_main.py --topology $topo_path --aggregation_strategy $aggregation_strategy --sample_alpha $sample_alpha --label_alpha $label_alpha --prox_coeff $prox_coeff --rounds 10
                done
            done
        done
    done
done
