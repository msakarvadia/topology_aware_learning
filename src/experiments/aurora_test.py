from __future__ import annotations

import logging
import pathlib
import os
import argparse
import sys
import time

import numpy as np
import torch

# from src.decentralized_app import DecentrallearnApp
from src.utils import process_futures_and_ckpt
from src.types import DataChoices
from src.experiments.parsl_setup import get_parsl_config
from pathlib import Path

import parsl
from parsl.app.app import python_app
from concurrent.futures import as_completed

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
        "--checkpoint_every",
        type=int,
        default=3,
        help="# of rounds to wait between checkpoints",
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
        "--train_test_val",
        type=float,
        default=None,
        nargs="+",
        help="Examples: '0.7, 0.2, 0.1' for train test val split, or '0.7, 0.3' for just trian test split",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate used for local training across all clients",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.0,
        help="momentum used for optimization",
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
        default=0,
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
        choices=[
            "mnist",
            "fmnist",
            "cifar10",
            "tiny_mem",
            # "tiny_mem_even_increment_one",
            "cifar10_augment",
            "cifar10_augment_vgg",
            "cifar10_vgg",
            "cifar10_mobile",
            "cifar10_vit",
            "cifar10_resnet18",
            "cifar10_resnet50",
            "cifar10_dropout",
            "cifar10_augment_dropout",
        ],
        help="Dataset (and corresponding model) to use",
    )
    parser.add_argument(
        "--tiny_mem_num_labels",
        type=int,
        default=50,
        choices=range(1, 101),
        metavar="[1-100]",
        help="# of different training data distributions to use for tiny mem",
    )
    parser.add_argument(
        "--aggregation_strategy",
        type=str,
        default="unweighted",
        choices=[
            "unweighted",
            "unweighted_fl",
            "weighted",
            "test_agg",
            "scale_agg",
            "degCent",
            "betCent",
            # "degCent_CA",
            # "betCent_CA",
            # "degCent_exp",
            # "betCent_exp",
            "cluster",
            "invCluster",
            "random",
        ],
        help="Type of aggregation stretegy used to among neighboring nodes.",
    )
    parser.add_argument(
        "--topology_file",
        type=str,
        default="../create_topo/topology/topo_1.txt",
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
        "--parsl_executor",
        type=str,
        default="experiment_per_node",
        choices=["experiment_per_node", "local", "node", "aurora_local"],
        help="Type of parsl executor to use. Local (local interactive job w/ 4 gpus), node (submitted to polaris nodes w/ 4 GPUs each)",
    )
    parser.add_argument(
        "--download",
        action="store_false",
        help="By default dataset is downloaded. If you set this flag, dataset will not be downloaded.",
    )
    parser.add_argument(
        "--no_train",
        action="store_false",
        help="By default flag is True and local training will be run. If you set this flag, then no-op version of this application will be performed where no training is done (used for debugging purposes).",
    )
    parser.add_argument(
        "--backdoor",
        action="store_true",
        help="By default flag is false and no backdoor will be set. If you set this flag, backdoor training will be performed..",
    )
    parser.add_argument(
        "--backdoor_proportion",
        type=float,
        default=0.1,
        help="Proportion of node local training data that is backdoored",
    )
    parser.add_argument(
        "--backdoor_node_idx",
        type=int,
        default=0,
        help="Node index in network with backdoored data",
    )
    parser.add_argument(
        "--random_bd",
        action="store_true",
        help="By default flag is false and backdoor triggers will be applied to the top left of an image. If you set this flag, backdoor triggers will randomly be set anywhere within an image.",
    )
    parser.add_argument(
        "--many_to_one",
        action="store_false",
        help="By default flag is true and all backdoored images will be assigned label=0. If you set this flag, each new label = (old_label + 1)% # of total labels (aka many to many labels). ",
    )
    parser.add_argument(
        "--offset_clients_data_placement",
        type=int,
        default=0,
        help="How many clients we want to offset data assignment by when assigning data to clients based on some metric",
    )
    parser.add_argument(
        "--centrality_metric_data_placement",
        type=str,
        default="degree",
        help="The centrality metric we want to use to assign data placement to the nodes by",
    )
    parser.add_argument(
        "--non_random_data_placement",
        action="store_false",
        help="By default flag is true, and data will be assigned randomly to nodes. If you set this flag, then data will be placed via the above specified centrality metric.",
    )
    parser.add_argument(
        "--softmax",
        action="store_true",
        help="By default flag is false, and aggregation coefficients will not be softmax (instead they will be normalized by dividing by sum of coefficients). If you set this flag, then aggregation coefficients will be normalized by a softmax.",
    )
    parser.add_argument(
        "--softmax_coeff",
        type=float,
        default=10,
        help="coefficient by which you scale your model weights before softmaxing them.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="decay parameter for exponential softmax coefficient scheduler.",
    )
    parser.add_argument(
        "--T_0",
        type=float,
        default=66,
        help="CA scheduler period.",
    )
    parser.add_argument(
        "--T_mult",
        type=float,
        default=1,
        help="CA scheduler arg.",
    )
    parser.add_argument(
        "--eta_min",
        type=float,
        default=1,
        help="CA scheduler arg.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=["exp", "CA"],
        help="Type of softmax coefficient scheduler",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["adam", "sgd", "adamw"],
        help="Type of optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--beta_1",
        type=float,
        default=0.9,
        help="param for AdamW optimizer.",
    )
    parser.add_argument(
        "--beta_2",
        type=float,
        default=0.98,
        help="param for AdamW optimizer.",
    )
    parser.add_argument(
        "--trigger",
        type=int,
        default=100,
        help="trigger for TinyMem BD",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=1000,
        help="TinyMem number of test data per task",
    )
    parser.add_argument(
        "--num_example",
        type=int,
        default=5000,
        help="TinyMem total # of data per task (test+ train)",
    )
    parser.add_argument(
        "--modulo",
        type=int,
        default=16381,
        help="TinyMem modulo applied to each # in seq",
    )
    parser.add_argument(
        "--length", type=int, default=20, help="TinyMem max # of numbers in each seq"
    )
    parser.add_argument(
        "--max_ctx", type=int, default=150, help="TinyMem max # of tokens in each seq"
    )
    parser.add_argument(
        "--n_layer", type=int, default=4, help="TinyMem max # of layer in model"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="multiply",
        choices=["multiply", "sum"],
        help="TinyMem max task type",
    )
    parser.add_argument(
        "--data_dis",
        type=str,
        default="evens",
        choices=["evens", "primes"],
        help="TinyMem data distribution type",
    )

    args = parser.parse_args()

    @python_app(executors=["experiment"])
    def run_experiment(args):
        from src.decentralized_app import DecentrallearnApp

        ### Parsl set up - TODO(MS): make parsl executor name an arg for polaris vs aurora
        import parsl
        from src.experiments.parsl_setup import get_parsl_config

        config, num_accelerators = get_parsl_config("aurora_single_experiment")
        parsl.load(config)
        ### Parsl set up

        decentral_app = DecentrallearnApp(
            rounds=args.rounds,
            dataset=args.dataset,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            data_dir=args.data_dir,
            topology_path=args.topology_file,
            # device=device,
            download=args.download,
            train=args.no_train,
            # test=args.no_test,
            label_alpha=args.label_alpha,
            sample_alpha=args.sample_alpha,
            participation=args.participation,
            seed=args.seed,
            log_dir=args.out_dir,
            aggregation_strategy=args.aggregation_strategy,
            prox_coeff=args.prox_coeff,
            train_test_val=(
                tuple(args.train_test_val) if args.train_test_val != None else None
            ),
            backdoor=args.backdoor,
            backdoor_proportion=args.backdoor_proportion,
            backdoor_node_idx=args.backdoor_node_idx,
            random_bd=args.random_bd,
            many_to_one=args.many_to_one,
            offset_clients_data_placement=args.offset_clients_data_placement,
            centrality_metric_data_placement=args.centrality_metric_data_placement,
            random_data_placement=args.non_random_data_placement,
            softmax=args.softmax,
            tiny_mem_num_labels=args.tiny_mem_num_labels,
            momentum=args.momentum,
            softmax_coeff=args.softmax_coeff,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            beta_1=args.beta_1,
            beta_2=args.beta_2,
            scheduler=args.scheduler,
            gamma=args.gamma,
            T_0=args.T_0,
            T_mult=args.T_mult,
            eta_min=args.eta_min,
            trigger=args.trigger,  # trigger for TinyMem BD
            num_test=args.num_test,  # TinyMem number of test data per task
            num_example=args.num_example,  # TinyMem total number of data per task (train + test)
            modulo=args.modulo,  # TinyMem modulo applied to each # in seq
            length=args.length,  # TinyMem max # of numbers in each seq
            max_ctx=args.max_ctx,  # TinyMem max # of tokens in each seq
            n_layer=args.n_layer,  # TinyMem # of layers in model
            task_type=args.task_type,  # TinyMem Task type: multiply | sum
            data_dis=args.data_dis,  # Tiny mem data distribution: primes | evens
        )
        # client_results = decentral_app.run()
        exit_value = decentral_app.run()
        parsl.dfk().cleanup()
        decentral_app.close()
        return exit_value

    # future = run_experiment(args)
    # print(f"{future=}")

    ######### Parsl
    config, num_accelerators = get_parsl_config(args.parsl_executor)

    parsl.load(config)
    #########

    start = time.time()
    future = run_experiment(args)
    print(f"Waiting on {future}")
    exit_value = future.result()
    end = time.time()
    print(f"{exit_value=}")
    print("Total time: ", end - start)
    parsl.dfk().cleanup()
