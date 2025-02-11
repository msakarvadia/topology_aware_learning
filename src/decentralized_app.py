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
import json

from src.decentralized_client import create_clients
from src.decentralized_client import create_centrality_dict
from src.decentralized_client import centrality_module_avg
from src.decentralized_client import unweighted_module_avg
from src.decentralized_client import weighted_module_avg
from src.decentralized_client import test_agg
from src.decentralized_client import scale_agg
from src.modules import create_model
from src.data import backdoor_data
from src.modules import load_data
from src.utils import load_checkpoint
from src.tasks import local_train
from src.tasks import no_local_train
from src.tasks import test_model
from src.types import DataChoices
from src.types import Result
from src.decentralized_client import DecentralClient
from src.aggregation_scheduler import CosineAnnealingWarmRestarts
from src.aggregation_scheduler import FakeScheduler
from src.aggregation_scheduler import ExponentialScheduler
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
        data_dir: str = "../data",
        topology_path: str = "topology/topo_1.txt",
        dataset: str = "mnist",
        rounds: int = 5,
        batch_size: int = 16,
        epochs: int = 2,
        lr: float = 1e-3,
        download: bool = False,
        train: bool = True,
        # test: bool = True,
        label_alpha: float = 100,
        sample_alpha: float = 100,
        participation: float = 1.0,
        seed: int | None = 0,
        log_dir: str = "./logs",
        aggregation_strategy: str = "weighted",
        prox_coeff: float = 0.1,
        train_test_val: tuple[int] = None,
        backdoor: bool = False,
        backdoor_proportion: float = 0.1,
        backdoor_node_idx: int = 0,
        random_bd: bool = False,
        # many-to-many or many-to-one backdoor from https://arxiv.org/pdf/1708.06733
        many_to_one: bool = True,
        offset_clients_data_placement: int = 0,  # this is how many clients we off set the data assignment by
        centrality_metric_data_placement: str = "degree",
        random_data_placement: bool = True,
        softmax: bool = False,  # this is if we normalize our weighting coefficients by softmax (or typical divide by sum)
        tiny_mem_num_labels: int = 50,
        momentum: float = 0,
        softmax_coeff: float = 10,
    ) -> None:

        # make the outdir
        args = locals()
        args.pop("self", None)
        args.pop("log_dir", None)
        args.pop("rounds", None)
        arg_path = "_".join(map(str, list(args.values())))

        # NOTE(MS): don't support backdoors in language modeling tasks
        if backdoor and dataset == "tiny_mem":
            raise ValueError("We don't support backdooring language modeling task")

        # Need to remove any . or / to ensure a single continuous file path
        arg_path = arg_path.replace(".", "")
        arg_path = arg_path.replace("/", "")
        self.run_dir = Path(f"{log_dir}/{arg_path}/")
        # check if run_dir exists, if not, make it
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        # Save args in the run_dir
        json.dump(args, open(f"{self.run_dir}/args.txt", "w"))

        if dataset == "mnist":
            self.dataset = DataChoices.MNIST
            self.num_labels = 10
        if dataset == "fmnist":
            self.dataset = DataChoices.FMNIST
            self.num_labels = 10
        if dataset == "cifar10":
            self.dataset = DataChoices.CIFAR10
            self.num_labels = 10
        if dataset == "cifar10_mobile":
            self.dataset = DataChoices.CIFAR10_MOBILE
            self.num_labels = 10
        if dataset == "cifar10_vit":
            self.dataset = DataChoices.CIFAR10_VIT
            self.num_labels = 10
        if dataset == "cifar10_resnet18":
            self.dataset = DataChoices.CIFAR10_RESTNET18
            self.num_labels = 10
        if dataset == "cifar10_resnet50":
            self.dataset = DataChoices.CIFAR10_RESTNET50
            self.num_labels = 10
        if dataset == "cifar10_augment":
            self.dataset = DataChoices.CIFAR10_AUGMENT
            self.num_labels = 10
        if dataset == "cifar10_augment_vgg":
            self.dataset = DataChoices.CIFAR10_AUGMENT_VGG
            self.num_labels = 10
        if dataset == "cifar10_vgg":
            self.dataset = DataChoices.CIFAR10_VGG
            self.num_labels = 10
        if dataset == "cifar10_dropout":
            self.dataset = DataChoices.CIFAR10_DROPOUT
            self.num_labels = 10
        if dataset == "cifar10_augment_dropout":
            self.dataset = DataChoices.CIFAR10_AUGMENT_DROPOUT
            self.num_labels = 10
        if dataset == "tiny_mem":
            self.dataset = DataChoices.TINYMEM
            self.num_labels = tiny_mem_num_labels

        # Initialize logging
        logging.basicConfig(
            filename=f"{self.run_dir}/experiment.log",
            encoding="utf-8",
            level=logging.DEBUG,
        )

        self.rng = numpy.random.default_rng(seed)
        self.seed = seed
        self.train_test_val = train_test_val
        # print(f"{train_test_val=}")
        if self.seed is not None:
            torch.manual_seed(seed)

        self.global_model = create_model(self.dataset)

        self.train = train
        # self.train, self.test = train, test
        self.train_data, self.test_data = None, None
        root = pathlib.Path(data_dir)
        self.train_data = load_data(
            self.dataset,
            root,
            train=True,
            download=True,
            tiny_mem_num_labels=tiny_mem_num_labels,
        )
        self.test_data = load_data(
            self.dataset,
            root,
            train=False,
            download=True,
            tiny_mem_num_labels=tiny_mem_num_labels,
        )

        self.backdoor = backdoor
        self.backdoor_proportion = backdoor_proportion
        self.backdoor_node_idx = backdoor_node_idx
        self.backdoor_test_data = None
        self.random_bd = random_bd
        self.many_to_one = many_to_one
        if self.backdoor:
            print("setting backdoor data")
            rng_seed = self.rng.integers(low=0, high=4294967295, size=1).item()
            self.test_data, self.backdoor_test_data = backdoor_data(
                dataset + "_test",
                self.test_data,
                self.test_data.targets,
                0.1,
                rng_seed,
                self.rng,
                self.num_labels,
                self.random_bd,
                self.many_to_one,
            )

        self.aggregation_strategy = aggregation_strategy
        self.centrality_metric = None
        self.softmax = softmax
        self.softmax_coeff = softmax_coeff
        self.aggregation_scheduler = FakeScheduler(self.softmax_coeff)
        if self.aggregation_strategy == "cluster":
            self.centrality_metric = "cluster"
            self.aggregation_function = centrality_module_avg
        if self.aggregation_strategy == "random":
            self.centrality_metric = "random"
            self.aggregation_function = centrality_module_avg
        if self.aggregation_strategy == "invCluster":
            self.centrality_metric = "invCluster"
            self.aggregation_function = centrality_module_avg
        if self.aggregation_strategy == "betCent":
            self.centrality_metric = "betweenness"
            self.aggregation_function = centrality_module_avg
        if self.aggregation_strategy == "betCent_exp":
            self.centrality_metric = "betweenness"
            self.aggregation_function = centrality_module_avg
            self.aggregation_scheduler = ExponentialScheduler(
                gamma=0.95,
                softmax_coeff=self.softmax_coeff,
            )
        if self.aggregation_strategy == "betCent_CA":
            self.centrality_metric = "betweenness"
            self.aggregation_function = centrality_module_avg
            self.aggregation_scheduler = CosineAnnealingWarmRestarts(
                T_0=66,
                T_mult=1,
                eta_min=1,
                last_round=-1,
                softmax_coeff=self.softmax_coeff,
            )
        if self.aggregation_strategy == "degCent":
            self.centrality_metric = "degree"
            self.aggregation_function = centrality_module_avg
        if self.aggregation_strategy == "degCent_exp":
            self.centrality_metric = "degree"
            self.aggregation_function = centrality_module_avg
            self.aggregation_scheduler = ExponentialScheduler(
                gamma=0.95,
                softmax_coeff=self.softmax_coeff,
            )
        if self.aggregation_strategy == "degCent_CA":
            self.centrality_metric = "degree"
            self.aggregation_function = centrality_module_avg
            self.aggregation_scheduler = CosineAnnealingWarmRestarts(
                T_0=20,
                T_mult=1,
                eta_min=1,
                last_round=-1,
                softmax_coeff=self.softmax_coeff,
            )
        if self.aggregation_strategy == "weighted":
            self.aggregation_function = weighted_module_avg
        if self.aggregation_strategy == "unweighted":
            self.aggregation_function = unweighted_module_avg
        if self.aggregation_strategy == "unweighted_fl":
            self.aggregation_function = unweighted_module_avg
        if self.aggregation_strategy == "test_agg":
            self.aggregation_function = test_agg
        if self.aggregation_strategy == "scale_agg":
            self.aggregation_function = scale_agg

        # NOTE (MS): Try assigning this in the job itself.
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum

        self.prox_coeff = prox_coeff
        self.participation = participation
        self.topology = numpy.loadtxt(topology_path, dtype=float)
        if self.aggregation_strategy == "unweighted_fl":
            self.topology = numpy.ones(self.topology.shape)
            ind = numpy.diag_indices(self.topology.shape[0])
            self.topology[ind[0], ind[1]] = torch.zeros(self.topology.shape[0])

        self.rounds = rounds
        self.start_round = 0  # this value will get overridden if we load from a ckpt

        if sample_alpha <= 0 or label_alpha <= 0:
            raise ValueError("Argument `alpha` must be greater than 0.")
        self.label_alpha = label_alpha
        self.sample_alpha = sample_alpha

        num_clients = self.topology.shape[0]
        if backdoor_node_idx >= num_clients:
            raise ValueError("Backdoor node index must be less than the # of clients.")

        self.offset_clients_data_placement = offset_clients_data_placement
        self.centrality_metric_data_placement = centrality_metric_data_placement
        self.random_data_placement = random_data_placement
        self.clients = create_clients(
            num_clients,
            self.dataset,
            self.train_data,
            self.num_labels,
            self.test_data,
            self.label_alpha,
            self.sample_alpha,
            self.rng,
            self.topology,
            self.prox_coeff,
            self.run_dir,
            self.train_test_val,
            self.backdoor_test_data,
            self.backdoor,
            self.backdoor_proportion,
            self.backdoor_node_idx,
            self.random_bd,
            self.many_to_one,
            self.offset_clients_data_placement,
            self.centrality_metric_data_placement,
            self.random_data_placement,
        )

        self.centrality_dict = create_centrality_dict(self.topology, self.rng)
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
    ) -> (
        list[Result],
        tuple(list[Result], DecentralClient),
        dict[int, dict[int, tuple(list[Result], DecentralClient)]],
    ):
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
            return self.client_results, [], self.round_states, self.run_dir

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
        return (
            self.client_results,
            train_result_futures,
            self.round_states,
            self.run_dir,
        )
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

            print(f"{client.idx=}")
            # print(f"{client.idx=}, {fed_prox_neighbors=}")
            # print(f"{self.dataset=}")
            future = job(
                train_input,
                round_idx,
                self.epochs,
                self.batch_size,
                self.lr,
                self.momentum,
                self.prox_coeff,
                # self.device,
                self.seed,
                self.backdoor,
                self.dataset,
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
                # pass in client futures even if they are not being aggregated
                # NOTE(MS): not sure if it makes sense to append futures in the event of a dropped client for a specific round?
                futures.append(agg_client)
                continue

            neighbor_idxs = client.get_neighbors()
            if len(neighbor_idxs) == 0:
                self.round_states[round_idx + 1][client.idx].update({"agg": agg_client})
                # pass in client futures even if they are not being aggregated
                futures.append(agg_client)
                continue

            # need to combine neighbors w/ client and pass to aggregate function
            agg_neighbors = []
            neighbor_idxs.append(client.idx)
            print(f"{neighbor_idxs=}")
            for i in neighbor_idxs:
                # NOTE (MS): we want to grab neighbors from the PRIOR round (as the current round still requires finishing)
                agg_neighbors.append(self.round_states[round_idx + 1][i]["train"])
            future = self.aggregation_function(
                agg_client,
                self.seed,
                *agg_neighbors,
                centrality_metric=self.centrality_metric,
                centrality_dict=self.centrality_dict,
                softmax=self.aggregation_scheduler.get_softmax_coeff(),
                # softmax=self.softmax,
                softmax_coeff=self.softmax_coeff,
            )
            futures.append(future)
            self.round_states[round_idx + 1][client.idx].update({"agg": future})
        self.aggregation_scheduler.step(round_idx)

        return futures  # results
