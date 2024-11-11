from __future__ import annotations

import logging
import pathlib
from pathlib import Path
from datetime import datetime
from concurrent.futures import as_completed

import numpy
import torch
import glob
import os

from src.decentralized_client import create_clients
from src.decentralized_client import unweighted_module_avg
from src.decentralized_client import weighted_module_avg
from src.modules import create_model
from src.modules import load_data
from src.modules import load_checkpoint
from src.modules import save_checkpoint
from src.tasks import local_train
from src.tasks import no_local_train
from src.tasks import test_model
from src.types import DataChoices
from src.types import Result
from src.decentralized_client import DecentralClient
from parsl.app.app import python_app

# Used within applications
APP_LOG_LEVEL = 21
logger = logging.getLogger(__name__)


class DecentrallearnApp:
    """Federated learning application.

    Args:
        clients: Number of simulated clients.
        rounds: Number of aggregation rounds to perform.
        dataset: Dataset (and corresponding model) to use.
        batch_size: Batch size used for local training across all clients.
        epochs: Number of epochs used during local training on all the clients.
        lr: Learning rate used during local training on all the clients.
        data_dir: Root directory where the dataset is stored or where
            you wish to download the data (i.e., `download=True`).
        device: Device to use for model training (e.g., `'cuda'`, `'cpu'`,
            `'mps'`).
        train: If `True` (default), the local training will be run. If `False,
            then a no-op version of the application will be performed where no
            training is done. This is useful for debugging purposes.
        test: If `True` (default), model testing is done at the end of each
            aggregation round.
        alpha: The number of data samples across clients is defined by a
            [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution)
            distribution. This value is used to define the uniformity of the
            amount of data samples across all clients. When data alpha
            is large, then the number of data samples across
            clients is uniform (default). When the value is very small, then
            the sample distribution becomes more non-uniform. Note: this value
            must be greater than 0.
        participation: The portion of clients that participate in an
            aggregation round. If set to 1.0, then all clients participate in
            each round; if 0.5 then half of the clients, and so on. At least
            one client will be selected regardless of this value and the
            number of clients.
        seed: Seed for reproducibility.
        run_dir: Run directory for results.
        topology: Network topology of each client's neighbors.
            Each client has a list of neighbor idxs.
    """

    def __init__(
        self,
        clients: int,
        rounds: int,
        dataset: DataChoices,
        num_labels: int,
        batch_size: int,
        epochs: int,
        lr: float,
        data_dir: pathlib.Path,
        topology: np.array,  # list[list[int]],
        device: str = "cpu",
        download: bool = False,
        train: bool = True,
        test: bool = True,
        label_alpha: float = 1e5,
        sample_alpha: float = 1e5,
        participation: float = 1.0,
        seed: int | None = None,
        run_dir: pathlib.Path = Path("./out"),
        aggregation_strategy: str = "weighted",
        prox_coeff: float = 0,
        checkpoint_every: int = 1,
    ) -> None:

        self.run_dir = run_dir

        # Initialize logging
        logging.basicConfig(
            filename=f"{self.run_dir}/experiment.log",
            encoding="utf-8",
            level=logging.DEBUG,
        )

        self.rng = numpy.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.dataset = dataset
        self.global_model = create_model(self.dataset)

        self.train, self.test = train, test
        self.train_data, self.test_data = None, None
        root = pathlib.Path(data_dir)
        if self.train:
            self.train_data = load_data(
                self.dataset,
                root,
                train=True,
                download=True,
            )
        if self.test:
            self.test_data = load_data(
                self.dataset,
                root,
                train=False,
                download=True,
            )

        self.aggregation_strategy = aggregation_strategy
        if self.aggregation_strategy == "weighted":
            self.aggregation_function = weighted_module_avg
        if self.aggregation_strategy == "unweighted":
            self.aggregation_function = unweighted_module_avg
        if self.aggregation_strategy == "parsl":
            self.aggregation_function = parsl_unweighted_module_avg

        self.device = torch.device(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_labels = num_labels
        self.checkpoint_every = checkpoint_every

        self.prox_coeff = prox_coeff
        self.participation = participation
        self.topology = topology

        self.rounds = rounds
        self.start_round = 0  # this value will get overridden if we load from a ckpt

        if sample_alpha <= 0 or label_alpha <= 0:
            raise ValueError("Argument `alpha` must be greater than 0.")
        self.label_alpha = label_alpha
        self.sample_alpha = sample_alpha

        self.clients = create_clients(
            clients,
            self.dataset,
            self.train,
            self.train_data,
            self.num_labels,
            self.test_data,
            self.label_alpha,
            self.sample_alpha,
            self.rng,
            self.topology,
            self.prox_coeff,
            self.run_dir,
        )
        logger.log(APP_LOG_LEVEL, f"Created {len(self.clients)} clients")

        self.client_results: list[Result] = []

        list_of_ckpts = glob.glob(f"{self.run_dir}/*.pth")
        if list_of_ckpts:
            # get the latest (most recently saved) ckpt
            checkpoint_path = max(list_of_ckpts, key=os.path.getctime)
            logger.log(
                APP_LOG_LEVEL, f"Loading lastest checkpoint from:  {checkpoint_path}"
            )
            self.start_round, self.clients, self.client_results = load_checkpoint(
                checkpoint_path, self.clients
            )
            self.start_round += 1  # we save the ckpt after the last round, so we add 1 to start the next round
            print(f"loaded latest ckpt from: {checkpoint_path}")

    def close(self) -> None:
        """Close the application."""
        pass

    def run(
        self,
    ) -> None:
        """Run the application.

        Args:

        Returns:
            List of results from each client after each round.
        """

        self.round_states = {}
        round_idx = self.start_round
        self.round_states[self.start_round] = {}
        for client_idx in range(len(self.clients)):
            self.round_states[self.start_round][client_idx] = {
                "agg": ([{}], self.clients[client_idx])
            }

        train_result_futures = []
        # this is to check if we are trying to resume training from a checkpoint that has already been completed
        if self.start_round >= self.rounds:
            # return []
            return self.client_results, [], self.round_states

        for round_idx in range(self.start_round, self.rounds):
            futures = self._federated_round(round_idx)
            train_result_futures.extend(futures)

        """
        checkpoint_path = f"{self.run_dir}/{round_idx}_ckpt.pth"
        resolved_futures = [i.result() for i in as_completed(train_result_futures)]
        [self.client_results.extend(i[0]) for i in resolved_futures]
        ckpt_clients = []
        for client_idx, client_future in self.round_states[round_idx + 1].items():
            result_object = client_future["agg"]
            print(f"{type(client_future['agg'][1])=}")
            # This is how we handle clients that are not returning appfutures (due to not being selected)
            if isinstance(result_object[1], DecentralClient):
                client = client_future["agg"][1]
            else:
                client = client_future["agg"].result()[1]
            ckpt_clients.append(client)
        save_checkpoint(round_idx, ckpt_clients, self.client_results, checkpoint_path)
        """
        return self.client_results, train_result_futures, self.round_states
        # return self.client_results

    def _federated_round(
        self,
        round_idx: int,
    ) -> list[Result]:
        """Perform a single round in federated learning.

        Specifically, this method runs the following steps:

        1. client selection
        2. local training
        3. model aggregation

        Args:
            round_idx: Round number.

        Returns:
            List of results from each client.
        """
        print("round idx: ", round_idx)
        job = local_train if self.train else no_local_train

        # select client indexes
        size = int(max(1, len(self.clients) * self.participation))
        assert 1 <= size <= len(self.clients)
        selected_client_idxs = self.rng.choice(
            list(range(len(self.clients))),
            size=size,
            replace=False,
        ).tolist()
        print(f"{selected_client_idxs=}")

        preface = f"({round_idx+1}/{self.rounds})"
        logger.log(
            APP_LOG_LEVEL,
            f"{preface} Starting local training for this round",
        )
        futures = []
        self.round_states[round_idx + 1] = {}
        for client in self.clients:
            train_input = self.round_states[round_idx][client.idx]["agg"]
            # only use clients for round if they are selected
            if client.idx not in selected_client_idxs:
                self.round_states[round_idx + 1][client.idx] = {"train": train_input}
                continue
            neighbor_idxs = client.get_neighbors()
            fed_prox_neighbors = []
            for i in neighbor_idxs:
                fed_prox_neighbors.append(self.round_states[round_idx][i]["agg"])
            future = job(
                train_input,
                round_idx,
                self.epochs,
                self.batch_size,
                self.lr,
                self.prox_coeff,
                self.device,
                *fed_prox_neighbors,
            )
            print(f"Launched Future: {future=}")
            self.round_states[round_idx + 1][client.idx] = {"train": future}

            preface = f"({round_idx+1}/{self.rounds}, client {client.idx}, )"
            logger.log(APP_LOG_LEVEL, f"{preface} Finished local training")

        for client in self.clients:
            agg_client = self.round_states[round_idx + 1][client.idx]["train"]
            # only use clients for round if they are selected
            if client.idx not in selected_client_idxs:
                self.round_states[round_idx + 1][client.idx].update({"agg": agg_client})
                continue

            neighbor_idxs = client.get_neighbors()
            if len(neighbor_idxs) == 0:
                continue
            # need to combine neighbors w/ client and pass to aggregate function
            agg_neighbors = []
            neighbor_idxs.append(client.idx)
            print(f"{neighbor_idxs=}")
            for i in neighbor_idxs:
                agg_neighbors.append(self.round_states[round_idx + 1][i]["train"])
            future = self.aggregation_function(agg_client, *agg_neighbors)
            futures.append(future)
            self.round_states[round_idx + 1][client.idx].update({"agg": future})

        return futures  # results
