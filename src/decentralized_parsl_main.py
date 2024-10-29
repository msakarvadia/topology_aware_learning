from __future__ import annotations

import logging
import pathlib
import os
import argparse

import numpy as np
import torch
import pandas as pd

from src.decentralized_app import DecentrallearnApp
from src.types import DataChoices
from pathlib import Path

import parsl
from parsl.config import Config

# PBSPro is the right provider for Polaris:
from parsl.providers import PBSProProvider

# The high throughput executor is for scaling to HPC systems:
from parsl.executors import HighThroughputExecutor

# address_by_interface is needed for the HighThroughputExecutor:
from parsl.addresses import address_by_interface

# You can use the MPI launcher, but may want the Gnu Parallel launcher, see below
from parsl.launchers import MpiExecLauncher, GnuParallelLauncher

if __name__ == "__main__":
    # set up arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="# of aggregation rounds",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size used for local training across all clients",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="# of epochs used during local training for each round on each client",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for reproducability",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate used for local training across all clients",
    )
    parser.add_argument(
        "--participation",
        type=float,
        default=1.0,
        help="""
            The portion of clients that participate in an
            aggregation round. If set to 1.0, then all clients participate in
            each round; if 0.5 then half of the clients, and so on. At least
            one client will be selected regardless of this value and the
            number of clients.
            """,
    )
    parser.add_argument(
        "--prox_coeff",
        type=float,
        default=0.1,
        help="""
            The proximal term coefficient. If this is set to 0, then no
            proximal term will be added.
            """,
    )
    parser.add_argument(
        "--sample_alpha",
        type=float,
        default=100,
        help="""
            The number of data samples across clients is defined by a
            [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution)
            distribution. This value is used to define the uniformity of the
            amount of data samples across all clients. When data alpha
            is large, then the number of data samples across
            clients is uniform (default). When the value is very small, then
            the sample distribution becomes more non-uniform. Note: this value
            must be greater than 0.
            """,
    )
    parser.add_argument(
        "--label_alpha",
        type=float,
        default=100,
        help="""
            The number of data samples across clients is defined by a
            [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution)
            distribution. This value is used to define the uniformity of the
            data labels across all clients. When data alpha
            is large, then the label distribution across
            clients is uniform (default). When the value is very small, then
            the label distribution becomes more non-uniform. Note: this value
            must be greater than 0.
            """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fmnist", "cifar10"],
        help="Dataset (and corresponding model) to use",
    )
    parser.add_argument(
        "--aggregation_strategy",
        type=str,
        default="unweighted",
        choices=["unweighted", "weighted"],
        help="Type of aggregation stretegy used to among neighboring nodes.",
    )
    parser.add_argument(
        "--topology_file",
        type=str,
        default="topology/topo_1.txt",
        help="Path to network topology saved as a numpy array adjacency matrix",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="logs",
        help="Path to output dir for all experiment log files/csvs",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data",
        help="Dataset (and corresponding model) to use",
    )
    parser.add_argument(
        "--download",
        action="store_false",
        help="By default dataset is downloaded. If you set this flag, dataset will not be downloaded.",
    )
    parser.add_argument(
        "--train",
        action="store_false",
        help="By default flag is True and local training will be run. If you set this flag, then no-op version of this application will be performed where no training is done (used for debugging purposes).",
    )
    parser.add_argument(
        "--test",
        action="store_false",
        help="By default flag is True and global model testing is done at end of each round. If you set this flag, then testing will not be performed.",
    )

    args = parser.parse_args()

    if args.dataset == "mnist":
        data = DataChoices.MNIST
        num_labels = 10
    if args.dataset == "fmnist":
        data = DataChoices.FMNIST
        num_labels = 10
    if args.dataset == "cifar10":
        data = DataChoices.CIFAR10
        num_labels = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set static out dir based on args
    # print("_".join(map(str, list(vars(args).values()))))
    arg_path = "_".join(map(str, list(vars(args).values())))
    # Need to remove any . or / to ensure a single continuous file path
    arg_path = arg_path.replace(".", "")
    arg_path = arg_path.replace("/", "")
    run_dir = Path(f"{args.out_dir}/{arg_path}/")
    # check if run_dir exists, if not, make it
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    topology = np.loadtxt(args.topology_file, dtype=float)
    # print(topology)
    clients = topology.shape[0]  # number of clients

    ######### Parsl
    run_dir = "/eagle/projects/argonne_tpc/mansisak/distributed_ml/src/"
    env = "/eagle/projects/argonne_tpc/mansisak/distributed_ml/env/"

    user_opts = {
        "worker_init": f"module use /soft/modulefiles; module load conda; conda activate {env}; cd {run_dir}",  # load the environment where parsl is installed
        "scheduler_options": "#PBS -l filesystems=home:eagle:grand",  # specify any PBS options here, like filesystems
        "account": "argonne_tpc",
        "queue": "debug",  # e.g.: "prod","debug, "preemptable" (see https://docs.alcf.anl.gov/polaris/running-jobs/)
        "walltime": "00:30:00",
        "nodes_per_block": 1,  # think of a block as one job on polaris, so to run on the main queues, set this >= 10
        # "cpus_per_node":    32, # Up to 64 with multithreading
        "available_accelerators": 4,  # Each Polaris node has 4 GPUs, setting this ensures one worker per GPU
        # "cores_per_worker": 8, # this will set the number of cpu hardware threads per worker.
    }

    config = Config(
        executors=[
            HighThroughputExecutor(
                label="Decentral_train",
                heartbeat_period=15,
                heartbeat_threshold=120,
                worker_debug=True,
                max_workers_per_node=4,
                available_accelerators=user_opts[
                    "available_accelerators"
                ],  # if this is set, it will override other settings for max_workers if set
                # cores_per_worker=user_opts["cores_per_worker"],
                address=address_by_interface("bond0"),
                cpu_affinity="block-reverse",
                prefetch_capacity=0,
                provider=PBSProProvider(
                    launcher=MpiExecLauncher(
                        bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"
                    ),
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
                    # cpus_per_node=user_opts["cpus_per_node"],
                    walltime=user_opts["walltime"],
                ),
            ),
        ],
        # run_dir=run_dir,
        checkpoint_mode="task_exit",
        retries=2,
        app_cache=True,
    )

    parsl.load(config)
    #########

    decentral_app = DecentrallearnApp(
        clients=clients,
        rounds=args.rounds,
        dataset=data,
        num_labels=num_labels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        data_dir=args.data_dir,
        topology=topology,
        device=device,
        download=args.download,
        train=args.train,
        test=args.test,
        label_alpha=args.label_alpha,
        sample_alpha=args.sample_alpha,
        participation=args.participation,
        seed=args.seed,
        run_dir=run_dir,
        aggregation_strategy=args.aggregation_strategy,
        prox_coeff=args.prox_coeff,
    )
    client_result = decentral_app.run()
    client_df = pd.DataFrame(client_result)
    client_df.to_csv(f"{run_dir}/client_stats.csv")
    print(client_df)

    decentral_app.close()
