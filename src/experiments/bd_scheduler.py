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
from src.create_topo.backdoor_topo import mk_backdoor_topos
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
            # "local",
            # "aurora_local",
            # "node",
        ],
        help="Type of parsl executor to use. experiment_per_node=Aurora, polaris_experiment_per_node=Polaris",
    )

    args = parser.parse_args()

    ######### Parsl
    config, num_accelerators = get_parsl_config(args.parsl_executor)

    parsl.load(config)
    #########

    start = time.time()
    param_list = []
    model_count = 0  # number of models in total created decentral Apps
    app_result_tuples = []
    num_experiments = 0

    for seed in [2, 1, 0]:
        paths, nodes = mk_backdoor_topos(num_nodes=4, seed=seed)
        for data in [
            "mnist",
            "fmnist",
            "tiny_mem",
            "cifar10_vgg",
            "cifar100_vgg",
        ]:
            wd = 0
            num_example = 5000
            checkpoint_every = 10
            task_type = "multiply"
            if data == "tiny_mem":
                num_example = 33000
                # num_example = 2000
                lr = 0.001
                # wd = 0.1
                optimizer = "adam"
                # optimizer = "adamw"
                # task_type = "sum"
            if data == "cifar10_vgg":
                lr = 0.0001
                optimizer = "adam"
                checkpoint_every = 5
            if data == "cifar100_vgg":
                lr = 0.0001
                optimizer = "adam"
                checkpoint_every = 1
            if data == "fmnist":
                lr = 0.01
                optimizer = "sgd"
                # optimizer = "adam"
            if data == "mnist":
                lr = 0.01
                optimizer = "sgd"
                # optimizer = "adam"
            # for softmax_coeff in [10, 100]:
            for softmax_coeff in [10]:
                # for softmax_coeff in [2, 4, 6, 8, 10, 100]:
                # iterate through aggregation strategies
                for aggregation_strategy in [
                    "unweighted",
                    "unweighted_fl",
                    "weighted",
                    "degCent",
                    "betCent",
                    "random",
                ]:
                    for scheduler in [None]:  # , "exp", "CA"]:
                        eta_min = 1
                        T_0 = 66
                        if scheduler == "CA":
                            eta_min = -50
                            T_0 = 10  # TODO this is worth varying between (5,8,10)
                        if scheduler == "CA" and (softmax_coeff in [2, 4, 6, 8]):
                            continue
                        if scheduler != None and (
                            aggregation_strategy
                            in ["unweighted", "weighted", "unweighted_fl", "random"]
                        ):
                            continue
                        # iterate through topologies
                        for topo, node_set in zip(paths, nodes):
                            # iterate through different backdoor node placements
                            print(f"{topo=}, {node_set=}")
                            topology = np.loadtxt(topo, dtype=float)
                            num_clients = topology.shape[0]

                            if softmax_coeff != 10 and (
                                aggregation_strategy
                                in ["unweighted", "weighted", "unweighted_fl"]
                            ):
                                continue

                            for client_idx in node_set:
                                # don't repeat unweighted fl at multiple bd placements
                                if aggregation_strategy == "unweighted_fl" and (
                                    client_idx != node_set[0]
                                ):
                                    continue

                                num_experiments += 1
                                # model_count += num_clients
                                experiment_args = {
                                    "dataset": data,
                                    "rounds": args.rounds,
                                    "topology_path": topo,
                                    "backdoor": True,
                                    "prox_coeff": 0,
                                    "epochs": 5,
                                    "backdoor_node_idx": client_idx,
                                    "aggregation_strategy": aggregation_strategy,
                                    "log_dir": "bd_scheduler_logs",
                                    "softmax": True,
                                    "optimizer": optimizer,
                                    "softmax_coeff": softmax_coeff,
                                    "sample_alpha": 1000,
                                    "label_alpha": 1000,
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
                                }

                                param_list.append(experiment_args)

    futures = [
        run_experiment(machine_name=args.parsl_executor, **experiment_args)
        for experiment_args in param_list
    ]

    print(f"{num_experiments=}")
    experiment_num = 0
    for future, args in zip(futures, param_list):
        print(f"Waiting for {future}")
        try:
            print(f"Got result {future.result()}")
        except Exception as e:
            print(f"Failing w/ exception: {e}")
            print(f"Details of failed experiment {experiment_num}:")
            print(args)
        experiment_num += 1

    # bc parsl workers keep having failures, lets try and just do a sleep while we wait for tasks
    # time.sleep(6 * 60 * 60)  # hours * 60 minutes * 60 seconds

    end = time.time()
    print("Total time: ", end - start)
    parsl.dfk().cleanup()
