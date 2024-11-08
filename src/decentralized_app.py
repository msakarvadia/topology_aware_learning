from __future__ import annotations

import logging
import pathlib
from pathlib import Path
from datetime import datetime

import numpy
import torch
import glob
import os
from concurrent.futures import as_completed

from src.decentralized_client import create_clients
from src.decentralized_client import unweighted_module_avg
from src.decentralized_client import weighted_module_avg
from src.decentralized_client import parsl_unweighted_module_avg
from src.modules import create_model
from src.modules import load_data
from src.modules import save_checkpoint
from src.modules import load_checkpoint
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
        job = local_train if self.train else no_local_train

        round_states = {}
        round_idx = 0
        round_states[round_idx] = {}
        for client_idx in range(len(self.clients)):
            round_states[round_idx][client_idx] = {
                "agg": ([{}], self.clients[client_idx])
            }

        # print(round_states)
        self.client_results = []
        train_result_futures = []
        for round_idx in range(self.start_round, self.rounds):
            print("round idx: ", round_idx)
            preface = f"({round_idx+1}/{self.rounds})"
            logger.log(
                APP_LOG_LEVEL,
                f"{preface} Starting local training for this round",
            )
            futures = []
            round_states[round_idx + 1] = {}
            for client_idx in range(len(self.clients)):
                print(
                    f"Keys before agg, {round_idx=}, {client_idx=}",
                    round_states[round_idx][client_idx].keys(),
                )
                train_input = round_states[round_idx][client_idx]["agg"]
                future = job(
                    train_input,
                    round_idx,
                    self.epochs,
                    self.batch_size,
                    self.lr,
                    self.prox_coeff,
                    self.device,
                )
                print(f"Launched Future: {future=}")
                futures.append(future)
                round_states[round_idx + 1][client_idx] = {"train": future}

                preface = f"({round_idx+1}/{self.rounds}, client {client_idx}, )"
                logger.log(APP_LOG_LEVEL, f"{preface} Finished local training")

            train_result_futures.extend(futures)

            for client in self.clients:
                neighbor_idxs = client.get_neighbors()
                if len(neighbor_idxs) == 0:
                    continue
                print(f"aggregating for {client.idx=} with {neighbor_idxs=}")
                # need to combine neighbors w/ client and pass to aggregate function
                # return client
                agg_client = round_states[round_idx + 1][client.idx]["train"]
                agg_neighbors = []
                neighbor_idxs.append(client.idx)
                for i in neighbor_idxs:
                    agg_neighbors.append(round_states[round_idx + 1][i]["train"])
                print(f"{agg_neighbors=}")
                print(f"{agg_client=}")
                future = self.aggregation_function(agg_neighbors, agg_client)
                round_states[round_idx + 1][client.idx].update({"agg": future})
                # round_states[round_idx + 1][client.idx].update(
                #    {"agg": round_states[round_idx + 1][client.idx]["train"]}
                # )

            print("next round of training")

            # train_result_future = self._federated_round(round_idx)
            # NOTE (MS): turning append not extending (might need to revisit this)
            # self.client_results.extend(train_result)

            """
            checkpoint_path = f"{self.run_dir}/{round_idx}_ckpt.pth"
            if round_idx % self.checkpoint_every == 0:
                resolved_futures = [
                    i.result()[0] for i in as_completed(train_result_futures)
                ]
                [self.client_results.extend(i) for i in resolved_futures]
                self.clients = [
                    i.result()[1] for i in as_completed(train_result_futures)
                ]
                print(self.clients)
                save_checkpoint(
                    round_idx, self.clients, self.client_results, checkpoint_path
                )
            """

        checkpoint_path = f"{self.run_dir}/{round_idx}_ckpt.pth"
        resolved_futures = [i.result() for i in as_completed(train_result_futures)]
        # resolved_futures = [i.result()[0] for i in as_completed(train_result_futures)]
        [self.client_results.extend(i[0]) for i in resolved_futures]
        self.clients = [i[1] for i in resolved_futures]
        save_checkpoint(round_idx, self.clients, self.client_results, checkpoint_path)
        print(self.client_results)
        return self.client_results  # , global_results

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
        results: list[Result] = []

        selected_clients = self.select_clients(
            self.participation, self.clients, self.rng
        )
        print(selected_clients)

        # TODO(MS): need to make variable
        # size = int(max(1, len(self.clients) * self.participation))
        num_clients = 3
        size = int(max(1, num_clients * self.participation))

        """
        assert 1 <= size <= len(self.clients)
        selected_clients = list(
            self.rng.choice(
                numpy.asarray(self.clients),
                size=size,
                replace=False,
            ),
        )
        futures =  self.launch_training_jobs(
            selected_clients,
            round_idx,
            self.epochs,
            self.batch_size,
            self.lr,
            self.prox_coeff,
            self.device,
        )
        """
        futures = []
        for i in range(size):
            client = selected_clients[i]
            print(f"TRAINING LOOP: {client=}")
            print(f"TRAINING LOOP: {client.result()=}")

            # for client in selected_clients:
            # neighbor_idxs = client.get_neighbors()
            # neighbors = numpy.asarray(self.clients)[neighbor_idxs].tolist()
            future = job(
                # result, client = job(
                client,
                round_idx,
                self.epochs,
                self.batch_size,
                self.lr,
                self.prox_coeff,
                self.device,
                self.clients,
                # neighbors,
            )
            print(f"Launched Future: {future=}")
            futures.append(future)
            preface = f"({round_idx+1}/{self.rounds}, client {client.idx}, )"
            logger.log(APP_LOG_LEVEL, f"{preface} Finished local training")

        print("next round of training")
        # each future is a tuple (results, client)
        # resolved_futures = [i.result() for i in as_completed(futures)]
        # assign results
        # [results.extend(i[0]) for i in resolved_futures]

        # assign clients
        clients = self.update_clients_from_futures(futures)
        print(f"in round: {self.clients=}")

        """
        for c in selected_clients:
            for client in self.clients:
                if c.idx == client.idx:
                    client.model.load_state_dict(c.model.state_dict())

        # assessing if reselecting clients has an effect
        selected_clients = list(
            self.rng.choice(
                numpy.asarray(self.clients),
                size=size,
                replace=False,
            ),
        )

        # aggregate for each client accross neighbors
        for client in selected_clients:
            # print(f"aggregating clients {self.clients=}")
            neighbor_idxs = client.get_neighbors()
            neighbors = numpy.asarray(self.clients)[neighbor_idxs].tolist()
            # skip aggregation for any client that has 0 neighbors in a given round
            if len(neighbors) == 0:
                continue
            avg_params = self.aggregation_function(neighbors)
            # avg_params = unweighted_module_avg(neighbors)
            client.model.load_state_dict(avg_params)
            preface = f"({round_idx+1}/{self.rounds}, client {client.idx}, )"
            logger.log(
                APP_LOG_LEVEL,
                f"{preface} Averaged the client's locally trained neighbors.",
            )
        """

        return futures  # results

    # TODO (MS) assign the threadpool executor: https://parsl.readthedocs.io/en/stable/faq.html#how-do-i-specify-where-apps-should-be-run
    @python_app(executors=["threadpool_executor"])
    def update_clients_from_futures(
        clients: list[DecentralClient],
        futures: tuple(list[Result], DecentralClient),
    ) -> list[DecentralClient]:
        # assign clients
        for i in futures:
            for client in clients:
                if client.idx == i[1].idx:
                    client.model.load_state_dict(i[1].model.state_dict())
        return clients

    @python_app(executors=["threadpool_executor"])
    def select_clients(
        participation: float,
        clients: list[DecentralClient],
        rng: numpy.random._generator.Generator,
    ) -> list[DecentralClient]:
        size = int(max(1, len(clients) * participation))
        assert 1 <= size <= len(clients)
        selected_clients = list(
            rng.choice(
                numpy.asarray(clients),
                size=size,
                replace=False,
            ),
        )
        return selected_clients

    @python_app(executors=["threadpool_executor"])
    def launch_training_jobs(
        selected_clients: list[DecentralClient],
        round_idx: int,
        epochs: int,
        batch_size: int,
        lr: float,
        prox_coeff: float,
        device: torch.device,
    ) -> list(tuple(list[Result], DecentralClient)):
        futures = []
        for client in selected_clients:
            future = tuple([{"place holder": 0}], client)
            """    
            future = job(
                client,
                round_idx,
                epochs,
                batch_size,
                lr,
                prox_coeff,
                device,
            )
            print(f"Launched Future: {future=}")
            """
            futures.append(future)
            preface = f"({round_idx+1}/{self.rounds}, client {client.idx}, )"
            logger.log(APP_LOG_LEVEL, f"{preface} Finished local training")

        return futures
