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
from src.create_topo.lm_topo import mk_lm_topos
from pathlib import Path

import parsl
from parsl.config import Config

# PBSPro is the right provider for Polaris:
from parsl.providers import PBSProProvider, LocalProvider

# The high throughput executor is for scaling to HPC systems:
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor

# address_by_interface is needed for the HighThroughputExecutor:
from parsl.addresses import address_by_interface

# You can use the MPI launcher, but may want the Gnu Parallel launcher, see below
from parsl.launchers import MpiExecLauncher, GnuParallelLauncher

if __name__ == "__main__":
    # set up arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="# of nodes per job",
    )
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
        choices=["local", "node"],
        help="Type of parsl executor to use. Local (local interactive job w/ 4 gpus), node (submitted to polaris nodes w/ 4 GPUs each)",
    )

    args = parser.parse_args()

    # this way we can guarantee clean checkpointing of desired # of rounds
    # NOTE (MS): this assert is not necessary, script will still work w/o it
    # just not guarantee that all rounds are completed
    assert args.rounds % args.checkpoint_every == 0

    ######### Parsl
    src_dir = "/eagle/projects/argonne_tpc/mansisak/distributed_ml/src/"
    env = "/eagle/projects/argonne_tpc/mansisak/distributed_ml/env/"

    user_opts = {
        "worker_init": f"module use /soft/modulefiles; module load conda; conda activate {env}; cd {src_dir}",  # load the environment where parsl is installed
        "scheduler_options": "#PBS -l filesystems=home:eagle:grand",  # specify any PBS options here, like filesystems
        "account": "argonne_tpc",
        "queue": "debug",  # e.g.: "prod","debug, "preemptable" (see https://docs.alcf.anl.gov/polaris/running-jobs/)
        "walltime": "01:00:00",
        "nodes_per_block": args.num_nodes,  # think of a block as one job on polaris, so to run on the main queues, set this >= 10
    }
    local_provider = LocalProvider(
        # 1 debug node
        nodes_per_block=user_opts["nodes_per_block"],
        init_blocks=1,
        min_blocks=0,
        max_blocks=1,  # Can increase more to have more parallel jobs
    )
    pbs_provider = PBSProProvider(
        launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"),
        account=user_opts["account"],
        queue=user_opts["queue"],
        select_options="ngpus=4",
        # PBS directives (header lines): for array jobs pass '-J' option
        scheduler_options=user_opts["scheduler_options"],
        # Command to be run before starting a worker, such as:
        worker_init=user_opts["worker_init"],
        # number of compute nodes allocated for each block
        nodes_per_block=user_opts["nodes_per_block"],
        init_blocks=1,
        min_blocks=0,
        max_blocks=1,  # Can increase more to have more parallel jobs
        walltime=user_opts["walltime"],
    )
    threadpool_executor = ThreadPoolExecutor(
        label="threadpool_executor",
        max_threads=2,
    )
    if args.parsl_executor == "local":
        executor = HighThroughputExecutor(
            label="decentral_train",
            heartbeat_period=15,
            heartbeat_threshold=120,
            worker_debug=True,
            max_workers_per_node=4,
            available_accelerators=4,
            # available_accelerators=["0", "1", "2", "3"],
            prefetch_capacity=0,
            provider=local_provider,
        )
    if args.parsl_executor == "node":
        executor = HighThroughputExecutor(
            label="decentral_train",
            heartbeat_period=15,
            heartbeat_threshold=120,
            worker_debug=True,
            max_workers_per_node=4,
            available_accelerators=4,
            # available_accelerators=["0", "1", "2", "3"],
            prefetch_capacity=0,
            provider=pbs_provider,
        )
    config = Config(
        executors=[executor, threadpool_executor],
        checkpoint_mode="task_exit",
        retries=2,
        app_cache=True,
    )

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
                for lr in [
                    0.01,
                ]:
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
                                for sample_alpha in [1, 10, 1000]:
                                    # Vary label heterogeneity
                                    for label_alpha in [1000]:  # [1, 10, 1000]:

                                        model_count += num_clients
                                        decentral_app = DecentrallearnApp(
                                            dataset=dataset,
                                            rounds=i,
                                            topology_path=topo,
                                            backdoor=False,
                                            prox_coeff=0,
                                            epochs=5,
                                            aggregation_strategy=aggregation_strategy,
                                            log_dir="lm_logs",
                                            sample_alpha=sample_alpha,
                                            label_alpha=label_alpha,
                                            softmax=True,
                                            softmax_coeff=softmax_coeff,
                                            tiny_mem_num_labels=50,
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

                                        if model_count > (args.num_nodes * 4):
                                            ######### Process and Save training results
                                            print(
                                                "There are more models than GPUs, so waiting for results, before making more experiments"
                                            )
                                            for result_tuple in app_result_tuples:
                                                process_futures_and_ckpt(*result_tuple)
                                            app_result_tuples = []
                                            model_count = 0

    end = time.time()
    print("Total time: ", end - start)
    parsl.dfk().cleanup()
    decentral_app.close()
