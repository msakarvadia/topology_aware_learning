from __future__ import annotations

import logging
import pathlib

import numpy
import torch

from src.fl_app import FedlearnApp
from src.types import DataChoices

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = "out"
    fl_app = FedlearnApp(
        clients=10,
        rounds=2,
        dataset=DataChoices.CIFAR10,
        batch_size=16,
        epochs=2,
        lr=1e-3,
        data_dir="../data",
        device=device,
        download=True,
        train=True,
        test=True,
        alpha=1e5,
        participation=1.0,
        seed=0,
    )
    fl_app.run(run_dir=run_dir)
    fl_app.close()
