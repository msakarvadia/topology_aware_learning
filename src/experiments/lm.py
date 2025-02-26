from __future__ import annotations

import logging
import pathlib
import os
import argparse
import sys
import time

import numpy as np
import torch
from pathlib import Path

from src.decentralized_app import DecentrallearnApp
from src.utils import process_futures_and_ckpt
from src.types import DataChoices
from src.create_topo.lm_topo import mk_lm_topos
from src.experiments.parsl_setup import get_parsl_config

import parsl

if __name__ == "__main__":
    # set up arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=1,
        help="# of rounds to wait between checkpoints (must be a factor of rounds)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=20,
        help="# of aggregation rounds (must be a multiple of checkpoint every)",
    )
    parser.add_argument(
        "--parsl_executor",
        type=str,
        default="local",
        choices=["local", "aurora_local", "node"],
        help="Type of parsl executor to use. Local (local interactive job w/ 4 gpus), node (submitted to polaris nodes w/ 4 GPUs each)",
    )

    args = parser.parse_args()

    # this way we can guarantee clean checkpointing of desired # of rounds
    # NOTE (MS): this assert is not necessary, script will still work w/o it
    # just not guarantee that all rounds are completed
    assert args.rounds % args.checkpoint_every == 0

    ######### Parsl
    config, num_accelerators = get_parsl_config(args.parsl_executor)

    parsl.load(config)
    #########

    paths = mk_lm_topos()

    start = time.time()
    # apps = {}
    model_count = 0  # number of models in total created decentral Apps
    for i in range(1, args.rounds + 1):
        # only submit job if round number is a multiple of checkpoint every
        if i % args.checkpoint_every == 0:
            print(f"running expeirment until round {i}")
            app_result_tuples = []
            for dataset in ["tiny_mem"]:
                for batch_size in [128]:  # [16, 128]:
                    for optimizer in ["sgd", "adam"]:  # [0.1, 0.01, 0.001]:
                        num_example = 5000
                        if optimizer == "sgd":
                            lr = 0.001
                            wd = 0
                        if optimizer == "adamw":
                            num_example = 2000
                            lr = 0.001
                            wd = 0.1
                        # for lr in [0.01, 0.001]:
                        for softmax_coeff in [10, 100]:
                            # iterate through aggregation strategies
                            for aggregation_strategy in [
                                "unweighted",
                                "weighted",
                                "degCent",
                                "betCent",
                                # "cluster",
                                # "random",
                                # "invCluster",
                            ]:
                                if softmax_coeff != 10 and (
                                    aggregation_strategy in ["unweighted", "weighted"]
                                ):
                                    continue
                                # iterate through topologies
                                for topo in paths:
                                    # iterate through different backdoor node placements
                                    print(f"{topo=}")
                                    topology = np.loadtxt(topo, dtype=float)
                                    num_clients = topology.shape[0]

                                    # Vary sample heterogeneity
                                    for sample_alpha in [1, 100, 1000]:
                                        # Vary label heterogeneity
                                        for label_alpha in [1000]:  # [1, 10, 1000]:

                                            print(
                                                f"{optimizer=}, {lr=}, {num_example=}"
                                            )
                                            model_count += num_clients
                                            decentral_app = DecentrallearnApp(
                                                dataset=dataset,
                                                rounds=i,
                                                topology_path=topo,
                                                prox_coeff=0,
                                                epochs=5,
                                                aggregation_strategy=aggregation_strategy,
                                                log_dir="lm_logs",
                                                sample_alpha=sample_alpha,
                                                label_alpha=label_alpha,
                                                softmax=True,
                                                softmax_coeff=softmax_coeff,
                                                tiny_mem_num_labels=5,
                                                lr=lr,
                                                batch_size=batch_size,
                                                optimizer=optimizer,
                                                weight_decay=wd,
                                                beta_1=0.9,
                                                beta_2=0.98,
                                                n_layer=1,
                                                backdoor=True,
                                                task_type="sum",
                                                num_example=num_example,
                                            )

                                            (
                                                client_results,
                                                train_result_futures,
                                                round_states,
                                                run_dir,
                                            ) = decentral_app.run()
                                            app_result_tuples.append(
                                                (
                                                    client_results,
                                                    train_result_futures,
                                                    round_states,
                                                    i,
                                                    run_dir,
                                                )
                                            )

                                            if model_count > (num_accelerators):
                                                ######### Process and Save training results
                                                print(
                                                    f"There are more models {model_count} than GPUs {num_accelerators}, so waiting for results, before making more experiments"
                                                )
                                                for result_tuple in app_result_tuples:
                                                    process_futures_and_ckpt(
                                                        *result_tuple
                                                    )
                                                app_result_tuples = []
                                                model_count = 0

    end = time.time()
    print("Total time: ", end - start)
    parsl.dfk().cleanup()
    decentral_app.close()
