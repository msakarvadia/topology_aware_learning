from __future__ import annotations

import logging
import pathlib
import os
import argparse
import sys
import time

import numpy as np
import torch

from src.decentralized_app import DecentrallearnApp
from src.utils import process_futures_and_ckpt
from src.types import DataChoices
from src.create_topo.ba_standard_topo import mk_ba_topos
from pathlib import Path

import parsl

# from parsl.app.app import python_app
from src.experiments.parsl_setup import get_parsl_config
from src.experiments.parsl_setup import run_experiment

if __name__ == "__main__":
    # set up arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="# of aggregation rounds (must be a multiple of checkpoint every)",
    )
    parser.add_argument(
        "--parsl_executor",
        type=str,
        default="experiment_per_node",
        choices=[
            "polaris_experiment_per_node",
            "experiment_per_node",
        ],
        help="Type of parsl executor to use. experiment_per_node=Aurora, polaris_experiment_per_node=Polaris",
    )

    args = parser.parse_args()

    start = time.time()
    param_list = []
    num_experiments = 0
    for seed in [
        0,
    ]:
        paths, nodes = mk_ba_topos(num_nodes=4, seed=seed)
        print(f"{nodes=}")
        print(f"{paths=}")
        for data in [
            "mnist",
            "fmnist",
            "tiny_mem",
            "cifar10_vgg",
            "cifar100_vgg",
            # "camelyon17",
            # "civilcomments",
            # "cifar10",
            # "cifar100",
        ]:
            wd = 0
            num_example = 5000
            checkpoint_every = 5
            task_type = "multiply"
            blur_level = 3
            noise_level = 5
            fog_level = 5
            epochs = 5
            ood_proportion = 0.1
            softmax_coeff = 10
            scheduler = None
            eta_min = 1
            T_0 = 66
            sample_alpha = 1000
            label_alpha = 1000
            ood_types = ["bd", "noise", "weather", "blur"]
            random_data_placement = True
            offset_clients = [
                0,
            ]
            if data == "tiny_mem":
                num_example = 33000
                lr = 0.001
                optimizer = "adam"
                ood_types = ["bd", "tiny_mem_7"]
            if data == "cifar10":
                optimizer = "sgd"
                lr = 0.001
                checkpoint_every = 5
            if data == "cifar100":
                optimizer = "sgd"
                lr = 0.001
                checkpoint_every = 5
            if data == "cifar10_vgg":
                lr = 0.0001
                optimizer = "adam"
                checkpoint_every = 5
                # NOTE(MS):
                # NOTE(MS):
                # blur_level = noise_level = fog_level = 1
                # NOTE(MS): vary local training epochs
                # epochs = 10
                # ood_proportion = 0.5
                ood_types = [
                    "bd",
                    "blur",
                ]  # "frost"]
            if data == "cifar100_vgg":
                lr = 0.0001
                optimizer = "adam"
                checkpoint_every = 5
                # NOTE(MS):
                # blur_level = noise_level = fog_level = 1
                # NOTE(MS): vary local training epochs
                # epochs = 10
                ## ood_proportion = 0.5
                ood_types = [
                    "bd",
                    "blur",
                ]  # "frost"]
            if data == "fmnist":
                lr = 0.01
                optimizer = "sgd"
            if data == "mnist":
                lr = 0.01
                optimizer = "sgd"
            if data == "camelyon17":
                lr = 0.001
                optimizer = "sgd"
                ood_types = [
                    "hospital",
                ]
                checkpoint_every = 2
            if data == "civilcomments":
                lr = 0.0005
                optimizer = "adamw"
                ood_types = [
                    None,
                ]
                # no ood_nodes
                paths = list(set(paths))
                nodes = [
                    [
                        0,
                    ],
                ] * len(paths)
                label_alpha = 1
                random_data_placement = False
                offset_clients = [
                    0,
                    1,
                    2,
                    3,
                ]
                checkpoint_every = 1

            for many_to_one in [True]:  # False
                # for softmax_coeff in [2, 4, 6, 8, 10, 100]:
                # iterate through aggregation strategies
                for aggregation_strategy in [
                    "unweighted",
                    "unweighted_fl",
                    # "closeCent",
                    # "eigenCent",
                    "degCent",
                    "betCent",
                    # "mhCent",
                    "weighted",
                    "random",
                    # "degCent_sim",
                    # "betCent_sim",
                ]:
                    for offset_clients_data_placement in offset_clients:
                        for ood_type in ood_types:
                            # for ood_type in ["bd", "weather", "blur", "noise"]:
                            # NOTE(MS): temp conditional to cut exp volume
                            # if "mnist" in data and (ood_type != "bd"):
                            #    continue
                            # if "mnist" in data and (ood_type == "bd"):
                            ood_proportion = 0.1
                            if ood_type == "bd":
                                ood_proportion = 0.02
                            # iterate through topologies
                            # NOTE(MS): here we do multi-node placements
                            for topo, node_set in zip(paths, nodes):
                                node_set = str(node_set).replace(" ", "")
                                topology = np.loadtxt(topo, dtype=float)
                                num_clients = topology.shape[0]

                                num_experiments += 1
                                experiment_args = {
                                    "dataset": data,
                                    "rounds": args.rounds,
                                    "topology_path": topo,
                                    "ood_type": ood_type,
                                    "prox_coeff": 0,
                                    "epochs": epochs,
                                    "ood_node_idxs": f"{node_set}",
                                    "aggregation_strategy": aggregation_strategy,
                                    "log_dir": "multi_node_ood_logs",
                                    "softmax": True,
                                    "optimizer": optimizer,
                                    "softmax_coeff": softmax_coeff,
                                    "sample_alpha": sample_alpha,
                                    "label_alpha": label_alpha,
                                    "lr": lr,
                                    "batch_size": 64,
                                    "weight_decay": wd,
                                    "beta_1": 0.9,
                                    "beta_2": 0.98,
                                    "n_layer": 1,
                                    "task_type": task_type,
                                    "num_example": num_example,
                                    "checkpoint_every": checkpoint_every,
                                    "tiny_mem_num_labels": 5,
                                    "scheduler": scheduler,
                                    "eta_min": eta_min,
                                    "T_0": T_0,
                                    "seed": seed,
                                    "ood_proportion": ood_proportion,
                                    "noise_level": noise_level,
                                    "blur_level": blur_level,
                                    "fog_level": fog_level,
                                    "many_to_one": many_to_one,
                                    "random_data_placement": random_data_placement,
                                    "offset_clients_data_placement": offset_clients_data_placement,  # this is how many clients we off set the data assignment by
                                }
                                param_list.append(experiment_args)

    print(f"{num_experiments=}")
    ######### Parsl
    config, num_accelerators = get_parsl_config(args.parsl_executor)

    parsl.load(config)
    #########
    futures = [
        run_experiment(machine_name=args.parsl_executor, **experiment_args)
        for experiment_args in param_list
    ]

    print(f"{num_experiments=}")
    experiment_num = 0
    for future, args in zip(futures, param_list):
        print(f"Waiting for {future}")
        try:
            print(f"Got result {future.result()} \n {args=}")
        except Exception as e:
            print(f"Failing w/ exception: {e}")
            print(f"Details of failed experiment {experiment_num}:")
            print(args)
            print(f"Exception type: {type(e).__name__}")
        experiment_num += 1

    end = time.time()
    print("Total time: ", end - start)
    parsl.dfk().cleanup()
