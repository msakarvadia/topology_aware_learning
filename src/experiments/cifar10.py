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
from src.create_topo.softmax_topo import mk_softmax_topos
from pathlib import Path

import parsl
from src.experiments.parsl_setup import get_parsl_config

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
    paths = mk_softmax_topos()

    start = time.time()
    # apps = {}
    model_count = 0  # number of models in total created decentral Apps
    for i in range(1, args.rounds + 1):
        # only submit job if round number is a multiple of checkpoint every
        if i % args.checkpoint_every == 0:
            print(f"running expeirment until round {i}")
            app_result_tuples = []
            for dataset in [
                "cifar10_vgg",
                # "cifar10_vit",
                # "cifar10_resnet18",
                # "cifar10_resnet50",
                # "cifar10_mobile",
            ]:
                for optimizer in ["sgd", "adam"]:  # , "adam"]:  # [0.1, 0.01, 0.001]:
                    if optimizer == "sgd":
                        lr = 0.001
                    if optimizer == "adam":
                        lr = 0.0001
                    # for lr in [0.001, 0.01]:  # [0.1, 0.01, 0.001]:
                    for backdoor in [True]:  # [0, 0.9]:
                        for softmax_coeff in [10, 100]:
                            # iterate through aggregation strategies
                            for aggregation_strategy in [
                                "unweighted",
                                "unweighted_fl",
                                "weighted",
                                "degCent",
                                "betCent",
                            ]:
                                if softmax_coeff != 100 and (
                                    aggregation_strategy
                                    in ["unweighted_fl", "unweighted", "weighted"]
                                ):
                                    continue

                                # iterate through topologies
                                for topo in paths:
                                    # iterate through different backdoor node placements
                                    print(f"{topo=}")
                                    """
                                    if "33_2" not in topo:
                                        print(
                                            "temp canceling experment to prune # of experiments"
                                        )
                                        continue
                                    """
                                    topology = np.loadtxt(topo, dtype=float)
                                    num_clients = topology.shape[0]

                                    # Vary sample heterogeneity
                                    for sample_alpha in [1000]:
                                        # Vary label heterogeneity
                                        for label_alpha in [1000]:  # [1, 10, 1000]:

                                            model_count += num_clients
                                            decentral_app = DecentrallearnApp(
                                                dataset=dataset,
                                                rounds=i,
                                                topology_path=topo,
                                                backdoor=backdoor,
                                                prox_coeff=0,
                                                epochs=5,
                                                aggregation_strategy=aggregation_strategy,
                                                log_dir="cifar10_logs",
                                                sample_alpha=sample_alpha,
                                                label_alpha=label_alpha,
                                                softmax=True,
                                                softmax_coeff=softmax_coeff,
                                                # momentum=momentum,
                                                lr=lr,
                                                optimizer=optimizer,
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
                                                    f"""There are more 
                                                    models {model_count} than GPUs {num_accelerators}, 
                                                    so waiting for results, before making more experiments"
                                                    """
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
