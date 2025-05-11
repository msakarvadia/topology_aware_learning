from __future__ import annotations

import logging
import pathlib
import os
import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch

from src.decentralized_app import DecentrallearnApp
from src.utils import process_futures_and_ckpt
from src.types import DataChoices
from src.create_topo.timing_topo import mk_timing_topos
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

    param_list = []
    model_count = 0  # number of models in total created decentral Apps
    app_result_tuples = []
    num_experiments = 0

    df = pd.DataFrame(columns=["num_clients", "time", "data", "trial"])
    # load dataframe
    time_trial_path = "experiment_time_trials.csv"
    if os.path.exists(time_trial_path):
        df = pd.read_csv(time_trial_path)
    for seed in [0, 1, 2]:
        paths, nodes = mk_timing_topos(num_nodes=1, seed=seed)
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
                lr = 0.001
                # checkpoint_every = 1
                optimizer = "adam"
            if data == "cifar10_vgg":
                lr = 0.0001
                optimizer = "adam"
                # checkpoint_every = 5
            if data == "cifar100_vgg":
                lr = 0.0001
                optimizer = "adam"
                # checkpoint_every = 1
            if data == "fmnist":
                lr = 0.01
                optimizer = "sgd"
            if data == "mnist":
                lr = 0.01
                optimizer = "sgd"
            for softmax_coeff in [10]:
                for aggregation_strategy in [
                    "unweighted",
                    # "unweighted_fl",
                    # "weighted",
                    # "degCent",
                    # "betCent",
                    # "random",
                ]:
                    for scheduler in [None]:  # , "exp", "CA"]:
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
                                num_experiments += 1
                                experiment_args = {
                                    "dataset": data,
                                    "rounds": args.rounds,
                                    "topology_path": topo,
                                    "backdoor": True,
                                    "prox_coeff": 0,
                                    "epochs": 5,
                                    "backdoor_node_idx": client_idx,
                                    "aggregation_strategy": aggregation_strategy,
                                    "log_dir": "timing_logs",
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
                                    # "eta_min": eta_min,
                                    # "T_0": T_0,
                                    "seed": seed,
                                }

                                start = time.time()
                                # add error handling to avoid repeat expeirments
                                experiment_df = df[
                                    (df.num_clients == num_clients)
                                    & (df.data == data)
                                    & (df.trial == seed)
                                ]
                                if not experiment_df.empty:
                                    # This means, experiment has already run
                                    continue
                                if data == "cifar100_vgg" and num_clients >= 64:
                                    continue
                                future = run_experiment(
                                    machine_name=args.parsl_executor, **experiment_args
                                )
                                print(f"Waiting for {future}")
                                try:
                                    print(f"Got result {future.result()}")
                                    end = time.time()
                                    total_time = end - start
                                    # record in a data frame
                                    df.loc[len(df)] = [
                                        num_clients,
                                        total_time,
                                        data,
                                        seed,
                                    ]
                                    # save dataframe
                                    df.to_csv(
                                        "/lus/flare/projects/AuroraGPT/mansisak/distributed_ml/src/experiments/experiment_time_trials.csv"
                                    )
                                except Exception as e:
                                    print(f"Failing w/ exception: {e}")
                                    print(args)
    parsl.dfk().cleanup()
