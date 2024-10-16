from __future__ import annotations

import logging
import pathlib

import numpy
import torch

from src.client import create_clients
from src.client import unweighted_module_avg
from src.modules import create_model
from src.modules import load_data
from src.tasks import local_train
from src.tasks import no_local_train
from src.tasks import test_model
from src.types import DataChoices
from src.types import Result

# Used within applications
APP_LOG_LEVEL = 21
logger = logging.getLogger(__name__)
# Initialize logging
logging.basicConfig(filename="example.log", encoding="utf-8", level=logging.DEBUG)


class FedlearnApp:
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
    """

    def __init__(
        self,
        clients: int,
        rounds: int,
        dataset: DataChoices,
        batch_size: int,
        epochs: int,
        lr: float,
        data_dir: pathlib.Path,
        device: str = "cpu",
        download: bool = False,
        train: bool = True,
        test: bool = True,
        alpha: float = 1e5,
        participation: float = 1.0,
        seed: int | None = None,
    ) -> None:

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

        self.device = torch.device(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.participation = participation

        self.rounds = rounds
        if alpha <= 0:
            raise ValueError("Argument `alpha` must be greater than 0.")
        self.alpha = alpha

        self.clients = create_clients(
            clients,
            self.dataset,
            self.train,
            self.train_data,
            self.alpha,
            self.rng,
        )
        logger.log(APP_LOG_LEVEL, f"Created {len(self.clients)} clients")

    def close(self) -> None:
        """Close the application."""
        pass

    def run(self, run_dir: pathlib.Path) -> None:
        """Run the application.

        Args:
            run_dir: Directory for run outputs.
        Returns:
            List of results from each client after each round.
        """
        results = []
        for round_idx in range(self.rounds):
            preface = f"({round_idx+1}/{self.rounds})"
            logger.log(
                APP_LOG_LEVEL,
                f"{preface} Starting local training for this round",
            )

            train_result = self._federated_round(round_idx, run_dir)
            results.extend(train_result)

            if self.test_data is not None:
                logger.log(
                    APP_LOG_LEVEL,
                    f"{preface} Starting the test for the global model",
                )
                test_result = test_model(
                    self.global_model,
                    self.test_data,
                    round_idx,
                    self.batch_size,
                    self.device,
                )
                # .result()
                logger.log(
                    APP_LOG_LEVEL,
                    f"{preface} Finished testing with test_loss="
                    f"{test_result['test_loss']:.3f}",
                )
        return results

    def _federated_round(
        self,
        round_idx: int,
        run_dir: pathlib.Path,
    ) -> list[Result]:
        """Perform a single round in federated learning.

        Specifically, this method runs the following steps:

        1. client selection
        2. local training
        3. model aggregation

        Args:
            round_idx: Round number.
            run_dir: Run directory for results.

        Returns:
            List of results from each client.
        """
        print("Round idx: ", round_idx)
        job = local_train if self.train else no_local_train
        # futures: list[TaskFuture[list[Result]]] = []
        results: list[Result] = []

        size = int(max(1, len(self.clients) * self.participation))
        assert 1 <= size <= len(self.clients)
        selected_clients = list(
            self.rng.choice(
                numpy.asarray(self.clients),
                size=size,
                replace=False,
            ),
        )

        for client in selected_clients:
            client.model.load_state_dict(self.global_model.state_dict())
            result = job(
                client,
                round_idx,
                self.epochs,
                self.batch_size,
                self.lr,
                self.device,
            )

            results.extend(result)

        preface = f"({round_idx+1}/{self.rounds})"
        logger.log(APP_LOG_LEVEL, f"{preface} Finished local training")
        avg_params = unweighted_module_avg(selected_clients)
        self.global_model.load_state_dict(avg_params)
        logger.log(
            APP_LOG_LEVEL,
            f"{preface} Averaged the returned locally trained models",
        )

        return results
