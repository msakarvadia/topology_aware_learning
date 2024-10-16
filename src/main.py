from __future__ import annotations

import logging
import pathlib
import os
import argparse

import numpy
import torch
import pandas as pd

from src.fl_app import FedlearnApp
from src.types import DataChoices
from pathlib import Path

if __name__ == "__main__":
    # set up arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clients",
        type=int,
        default=3,
        help="# of simulated clients",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
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
        "--alpha",
        type=float,
        default=1,
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
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fmnist", "cifar10"],
        help="Dataset (and corresponding model) to use",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./out",
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
    if args.dataset == "fmnist":
        data = DataChoices.FMNIST
    if args.dataset == "cifar10":
        data = DataChoices.CIFAR10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.out_dir)
    # check if run_dir exists, if not, make it
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    fl_app = FedlearnApp(
        clients=args.clients,
        rounds=args.rounds,
        dataset=data,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        data_dir=args.data_dir,
        device=device,
        download=args.download,
        train=args.train,
        test=args.test,
        alpha=args.alpha,
        participation=args.participation,
        seed=args.seed,
        run_dir=run_dir,
    )
    client_result, global_result = fl_app.run()
    client_df = pd.DataFrame(client_result)
    client_df.to_csv(f"{run_dir}/client_stats.csv")
    print(client_df)

    global_df = pd.DataFrame(global_result)
    global_df.to_csv(f"{run_dir}/global_stats.csv")
    print(global_df)

    fl_app.close()
