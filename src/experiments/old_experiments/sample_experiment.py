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
from pathlib import Path

import parsl
from src.experiments.parsl_setup import get_parsl_config

if __name__ == "__main__":
    # set up arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=5,
        help="# of rounds to wait between checkpoints (must be a factor of rounds)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
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

    start = time.time()
    for i in range(1, args.rounds + 1):
        # only submit job if round number is a multiple of checkpoint every
        if i % args.checkpoint_every == 0:
            print(f"running expeirment until round {i}")
            # begin experiment
            app_result_tuples = []
            for lr in [0.1, 0.01, 0.001]:
                for epochs in [5, 10, 15]:
                    # iterate through aggregation strategies
                    for data in [
                        "cifar10",
                        "cifar10_augment",
                        "cifar10_augment_vgg",
                        "cifar10_vgg",
                        "cifar10_dropout",
                        "cifar10_augment_dropout",
                    ]:  # , 'cifar100']:
                        for aggregation_strategy in [
                            "unweighted",
                            "weighted",
                            "degCent",
                            "betCent",
                            # "cluster",
                            # "invCluster",
                        ]:
                            for topo in [
                                # "../create_topo/topology/topo_1.txt",
                                # "../create_topo/topology/topo_2.txt",
                                # "../create_topo/topology/topo_3.txt",
                                # "../create_topo/topology/topo_4.txt",
                                "../create_topo/topology/topo_5.txt",  # NOTE(MS): has floating nodes
                                # "../create_topo/topology/topo_6.txt",
                                # "../create_topo/topology/topo_7.txt",
                            ]:
                                decentral_app = DecentrallearnApp(
                                    rounds=i,
                                    topology_path=topo,
                                    prox_coeff=0,
                                    epochs=epochs,
                                    backdoor=False,
                                    aggregation_strategy=aggregation_strategy,
                                    softmax=True,
                                    dataset=data,
                                    momentum=0.9,
                                    lr=lr,
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

            ######### Process and Save training results
            for result_tuple in app_result_tuples:
                process_futures_and_ckpt(*result_tuple)

    end = time.time()
    print("Total time: ", end - start)
    parsl.dfk().cleanup()
    decentral_app.close()
