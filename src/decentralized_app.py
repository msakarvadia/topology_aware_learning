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
import sys
import shutil

from src.decentralized_client import create_clients
from src.decentralized_client import create_centrality_dict
from src.decentralized_client import centrality_module_avg
from src.decentralized_client import updated_centrality_module_avg
from src.decentralized_client import sim_centrality_module_avg
from src.decentralized_client import unweighted_module_avg
from src.decentralized_client import weighted_module_avg
from src.decentralized_client import test_agg
from src.decentralized_client import scale_agg
from src.decentralized_client import update_random_agg_coeffs
from src.modules import create_model
from src.data import ood_data
from src.modules import load_data
from src.utils import load_checkpoint
from src.utils import get_client_aggregation_weights
from src.utils import get_adj_mat
from src.tasks import local_train
from src.tasks import no_local_train
from src.tasks import test_model
from src.types import DataChoices
from src.types import Result
from src.decentralized_client import DecentralClient
from src.aggregation_scheduler import CosineAnnealingWarmRestarts
from src.aggregation_scheduler import BaseScheduler
from src.aggregation_scheduler import ExponentialScheduler
from src.aggregation_scheduler import OscilateScheduler

# from parsl.app.app import python_app
from src.utils import process_futures_and_ckpt
from src.utils import set_file_logger

# Used within applications
APP_LOG_LEVEL = 21
logger = logging.getLogger("decentral_app")

# parsl_logger = logging.getLogger("parsl")
# parsl_logger.setLevel(logging.INFO)

# Optionally set a file handler
# fh = logging.FileHandler("parsl_custom.log")
# fh.setLevel(logging.DEBUG)
# parsl_logger.addHandler(fh)


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
        ood_type: str = None,  # bd, weather, blur, noise, digital
        # backdoor: bool = False,
        ood_proportion: float = 0.1,
        ood_node_idxs: list[int] = [
            0,
        ],
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
        optimizer: str = "sgd",
        weight_decay: float = 0,
        beta_1: float = 0.9,
        beta_2: float = 0.98,
        scheduler: str = None,  # Exp, CA
        gamma: float = 0.95,
        T_0: float = 66,
        T_mult: float = 1,
        eta_min: float = 1,
        trigger: int = 100,  # trigger for TinyMem BD
        num_test: int = 1000,  # TinyMem number of test data per task
        num_example: int = 5000,  # TinyMem total number of data per task (train + test)
        modulo: int = 16381,  # TinyMem modulo applied to each # in seq
        length: int = 20,  # TinyMem max # of numbers in each seq
        max_ctx: int = 150,  # TinyMem max # of tokens in each seq
        n_layer: int = 4,  # TinyMem # of layers in model
        task_type: str = "multiply",  # TinyMem Task type: multiply | sum
        data_dis: str = "evens",  # Tiny mem data dir: primes | evens
        checkpoint_every: int = 5,  # checkpoint every X rounds
        frob_radius: float = 2,  # frobenius radius for double stoch
        matrix_type: str = "row_stoch",  # row_stoch vs. double stoch
        sink_temp: float = 0.01,  # HP for sinkhorn_knopp for double stoch (not high impact)
        blur_level: int = 3,  # HP for blur corruption
        noise_level: int = 5,  # HP for noise curroption
        fog_level: int = 5,  # HP for fog corruption
    ) -> None:

        # make the outdir
        args = locals()
        args["topology_path"] = os.path.basename(args["topology_path"])
        args.pop("self", None)
        args.pop("log_dir", None)
        args.pop("rounds", None)
        args.pop("checkpoint_every", None)
        arg_path = "_".join(map(str, list(args.values())))

        # Need to remove any . or / to ensure a single continuous file path
        arg_path = arg_path.replace(".", "")
        arg_path = arg_path.replace(",", "_")
        arg_path = arg_path.replace("[", "")
        arg_path = arg_path.replace("]", "")
        arg_path = arg_path.replace("/", "")
        self.run_dir = Path(f"{log_dir}/{arg_path}/")
        # check if run_dir exists, if not, make it
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        # Save args in the run_dir
        json.dump(args, open(f"{self.run_dir}/args.txt", "w"))

        if dataset == "civilcomments":
            self.dataset = DataChoices.CIVILCOMMENTS
            self.num_labels = 2
        if dataset == "camelyon17":
            self.dataset = DataChoices.CAMELYON17
            self.num_labels = 2
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
        if dataset == "cifar100_vgg":
            self.dataset = DataChoices.CIFAR100_VGG
            self.num_labels = 100
        if dataset == "cifar10_dropout":
            self.dataset = DataChoices.CIFAR10_DROPOUT
            self.num_labels = 10
        if dataset == "cifar10_augment_dropout":
            self.dataset = DataChoices.CIFAR10_AUGMENT_DROPOUT
            self.num_labels = 10
        if dataset == "tiny_mem":
            self.dataset = DataChoices.TINYMEM
            self.num_labels = tiny_mem_num_labels
        if dataset == "tiny_mem_even_increment_one":
            self.dataset = DataChoices.TINYMEM_EVEN_INCREMENT_ONE
            self.num_labels = tiny_mem_num_labels

        # Initialize logging
        logger = set_file_logger(
            filename=f"{self.run_dir}/experiment.log",
            name="decentral_app",
        )
        """
        logging.basicConfig(
            filename=f"{self.run_dir}/experiment.log",
            encoding="utf-8",
            level=logging.DEBUG,
        )
        """

        self.rng = numpy.random.default_rng(seed)
        self.seed = seed
        self.train_test_val = train_test_val
        if self.seed is not None:
            torch.manual_seed(seed)

        self.max_ctx = max_ctx
        self.n_layer = n_layer

        self.checkpoint_every = checkpoint_every
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
            trigger=trigger,
            num_test=num_test,
            num_example=num_example,
            modulo=modulo,
            max_ctx=max_ctx,
            task_type=task_type,
            data_dis=data_dis,  # Tiny mem data distribution: primes | evens
            length=length,
        )
        self.test_data = load_data(
            self.dataset,
            root,
            train=False,
            download=True,
            tiny_mem_num_labels=tiny_mem_num_labels,
            trigger=trigger,
            num_test=num_test,
            num_example=num_example,
            modulo=modulo,
            max_ctx=max_ctx,
            task_type=task_type,
            data_dis=data_dis,  # Tiny mem data distribution: primes | evens
            length=length,
        )

        self.topology = numpy.loadtxt(topology_path, dtype=float)
        num_clients = self.topology.shape[0]

        self.blur_level = blur_level
        self.noise_level = noise_level
        self.fog_level = fog_level
        self.ood_type = ood_type
        self.ood_proportion = ood_proportion
        self.ood_node_idxs = eval(ood_node_idxs)
        self.ood_test_data = None
        self.random_bd = random_bd
        self.many_to_one = many_to_one

        self.offset_clients_data_placement = offset_clients_data_placement
        self.centrality_metric_data_placement = centrality_metric_data_placement
        self.random_data_placement = random_data_placement
        self.trigger = trigger
        if self.ood_type:
            logger.log(APP_LOG_LEVEL, f"setting backdoor data")
            rng_seed = self.rng.integers(low=0, high=4294967295, size=1).item()
            self.test_data, self.ood_test_data = ood_data(
                dataset + "_test",
                self.test_data,
                self.test_data.targets,
                0.1,
                rng_seed,
                self.rng,
                self.num_labels,
                self.random_bd,
                self.many_to_one,
                # for propoer checkpointing purposes we need to save some additional info
                # self.offset_clients_data_placement,
                # self.centrality_metric_data_placement,
                # self.random_data_placement,
                # self.ood_node_idx,
                # num_clients=num_clients,
                # test_data=1,  # this is trianing data
                trigger=self.trigger,
                ood_type=self.ood_type,
                blur_level=self.blur_level,
                noise_level=self.noise_level,
                fog_level=self.fog_level,
            )

        self.aggregation_strategy = aggregation_strategy
        self.centrality_metric = None
        self.softmax = softmax
        self.softmax_coeff = softmax_coeff
        self.aggregation_scheduler = BaseScheduler(self.softmax_coeff)
        if self.aggregation_strategy == "betCent_sim":
            self.centrality_metric = "betweenness"
            self.aggregation_function = sim_centrality_module_avg
        if self.aggregation_strategy == "degCent_sim":
            self.centrality_metric = "degree"
            self.aggregation_function = sim_centrality_module_avg
        if self.aggregation_strategy == "random":
            self.centrality_metric = "random"
            self.aggregation_function = centrality_module_avg
        if self.aggregation_strategy == "betCent":
            self.centrality_metric = "betweenness"
            self.aggregation_function = updated_centrality_module_avg
        if self.aggregation_strategy == "degCent":
            self.centrality_metric = "degree"
            self.aggregation_function = updated_centrality_module_avg
        if self.aggregation_strategy == "closeCent":
            self.centrality_metric = "closeness"
            self.aggregation_function = updated_centrality_module_avg
        if self.aggregation_strategy == "eigenCent":
            self.centrality_metric = "eigen"
            self.aggregation_function = updated_centrality_module_avg
        if self.aggregation_strategy == "mhCent":
            self.centrality_metric = "metro_hast"
            self.aggregation_function = updated_centrality_module_avg
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

        self.frob_radius = frob_radius
        self.matrix_type = matrix_type
        self.sink_temp = sink_temp
        self.centrality_dict = create_centrality_dict(self.topology, self.rng)

        logger.log(APP_LOG_LEVEL, f"{self.centrality_metric}")
        self.adj_mat = get_adj_mat(
            centrality_metric=self.centrality_metric,
            softmax_coeff=self.aggregation_scheduler.get_softmax_coeff(),
            softmax_bool=self.softmax,
            matrix_type=self.matrix_type,
            topology=self.topology,
            centrality_dict=self.centrality_dict,
            R=self.frob_radius,
            sink_temp=self.sink_temp,
        )
        if scheduler == "CA":
            self.aggregation_scheduler = CosineAnnealingWarmRestarts(
                T_0=T_0,
                T_mult=T_mult,
                eta_min=eta_min,
                last_round=-1,
                softmax_coeff=self.softmax_coeff,
            )
        if scheduler == "exp":
            self.aggregation_scheduler = ExponentialScheduler(
                gamma=gamma,
                softmax_coeff=self.softmax_coeff,
            )
        if scheduler == "osc":
            self.aggregation_scheduler = OscilateScheduler(
                T_0=T_0,
                softmax_coeff=self.softmax_coeff,
            )
        # NOTE (MS): Try assigning this in the job itself.
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.prox_coeff = prox_coeff
        self.participation = participation
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

        if max(self.ood_node_idxs) >= num_clients:
            raise ValueError("Backdoor node index must be less than the # of clients.")

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
            self.ood_test_data,
            self.ood_type,
            self.ood_proportion,
            self.ood_node_idxs,
            self.random_bd,
            self.many_to_one,
            self.offset_clients_data_placement,
            self.centrality_metric_data_placement,
            self.random_data_placement,
            self.run_dir,
            self.trigger,
            self.blur_level,
            self.noise_level,
            self.fog_level,
        )

        logger.log(APP_LOG_LEVEL, f"Created {len(self.clients)} clients")

        self.client_results: list[Result] = []

        logger.log(APP_LOG_LEVEL, f"{self.run_dir=}")
        logger.log(APP_LOG_LEVEL, f"{os.listdir(self.run_dir)=}")
        list_of_ckpts = glob.glob(f"{self.run_dir}/*.pth")
        if list_of_ckpts:
            # get the latest (most recently saved) ckpt
            checkpoint_path = max(list_of_ckpts, key=os.path.getctime)
            logger.log(
                APP_LOG_LEVEL, f"Loading lastest checkpoint from:  {checkpoint_path}"
            )
            try:
                (
                    self.start_round,
                    self.clients,
                    self.client_results,
                    self.aggregation_scheduler,
                ) = load_checkpoint(
                    checkpoint_path, self.clients, self.aggregation_scheduler
                )
                logger.log(APP_LOG_LEVEL, f"ckpt loaded from: {checkpoint_path}")
                logger.log(APP_LOG_LEVEL, f"{len(self.client_results)=}")

            except Exception as error:
                logger.log(
                    APP_LOG_LEVEL, f"ERROR: Corrupted checkpoint:  {checkpoint_path}"
                )
                logger.log(APP_LOG_LEVEL, f"{error}")
                os.remove(checkpoint_path)
                # 2 error means corrupted ckpt
                return 2
            self.start_round += 1  # we save the ckpt after the last round, so we add 1 to start the next round
            logger.log(APP_LOG_LEVEL, f"loaded latest ckpt from: {checkpoint_path}")

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
            logger.log(
                APP_LOG_LEVEL,
                f"{self.start_round=}, {self.rounds=}; resuming from ckpt that has already been completed...returning",
            )
            # return []
            return 0
            # return self.client_results, [], self.round_states, self.run_dir

        for round_idx in range(self.start_round, self.rounds):
            preface = f"({round_idx+1}/{self.rounds})"
            futures = self._federated_round(round_idx)
            train_result_futures.extend(futures)
            # save a checkpoint here
            if (round_idx % self.checkpoint_every == 0) and (round_idx != 0):
                logger.log(
                    APP_LOG_LEVEL,
                    f"INTERIM: {preface} {len(futures)=}, {len(train_result_futures)=}, {len(self.client_results)=}",
                )
                logger.log(
                    APP_LOG_LEVEL,
                    f"INTERIM: {preface} Attempting to save ckpt",
                )
                try:
                    process_futures_and_ckpt(
                        self.client_results,
                        train_result_futures,
                        self.round_states,
                        round_idx,
                        self.run_dir,
                    )
                except Exception as e:
                    # failed to save ckpt file
                    logger.log(
                        APP_LOG_LEVEL,
                        f"{e}",
                    )
                    return 2
                logger.log(
                    APP_LOG_LEVEL,
                    f"INTERIM: {preface} Have saved a ckpt",
                )

            # if an round -1 key is in round_states dict, delete it
            old_round = round_idx - 1
            if old_round in self.round_states:
                del self.round_states[round_idx - 1]

        logger.log(
            APP_LOG_LEVEL,
            f"{preface} {len(futures)=}, {len(train_result_futures)=}, {len(self.client_results)=}",
        )
        logger.log(
            APP_LOG_LEVEL,
            f"{preface} Attempting to save ckpt",
        )
        try:
            process_futures_and_ckpt(
                self.client_results,
                train_result_futures,
                self.round_states,
                self.rounds,
                self.run_dir,
            )
        except Exception as e:
            logger.log(
                APP_LOG_LEVEL,
                f"{e}",
            )
            return 2
        logger.log(
            APP_LOG_LEVEL,
            f"{preface} Have saved a ckpt",
        )

        # NOTE(MS): how would we do parsl clean up?
        # parsl.dfk().cleanup()
        return 0

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
        job = local_train if self.train else no_local_train

        # select client indexes
        size = int(max(1, len(self.clients) * self.participation))
        assert 1 <= size <= len(self.clients)
        selected_client_idxs = self.rng.choice(
            list(range(len(self.clients))),
            size=size,
            replace=False,
        ).tolist()

        preface = f"({round_idx+1}/{self.rounds})"
        logger.log(
            APP_LOG_LEVEL,
            f"""{preface} Starting local training for {round_idx}, {selected_client_idxs=}""",
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
                self.momentum,
                self.prox_coeff,
                # self.device,
                self.seed,
                self.ood_type,
                self.dataset,
                self.optimizer,
                self.weight_decay,
                self.beta_1,
                self.beta_2,
                *fed_prox_neighbors,
            )
            self.round_states[round_idx + 1][client.idx] = {"train": future}

            preface = f"({round_idx+1}/{self.rounds}, client {client.idx}, )"
            logger.log(APP_LOG_LEVEL, f"{preface} Got 'future' for local training")

        # need to update random centrality metric before each round
        if self.centrality_metric == "random":
            self.centrality_dict = update_random_agg_coeffs(
                seed=self.seed,
                round_idx=round_idx,
                num_clients=len(self.clients),
                centrality_dict=self.centrality_dict,
            )

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
            logger.log(APP_LOG_LEVEL, f"{neighbor_idxs=}, {self.adj_mat=}")
            for i in neighbor_idxs:
                # NOTE (MS): we want to grab neighbors from the PRIOR round (as the current round still requires finishing)
                agg_neighbors.append(self.round_states[round_idx + 1][i]["train"])

            weights = get_client_aggregation_weights(
                adj_mat=self.adj_mat,
                client_idx=client.idx,
                neighbor_idxs=neighbor_idxs,
            )
            logger.log(APP_LOG_LEVEL, f"***************{client.idx=}, {weights=}")
            future = self.aggregation_function(
                agg_client,
                self.seed,
                *agg_neighbors,
                centrality_metric=self.centrality_metric,
                centrality_dict=self.centrality_dict,
                softmax=self.softmax,
                # softmax_coeff=self.softmax_coeff,
                softmax_coeff=self.aggregation_scheduler.get_softmax_coeff(),
                weights=weights,
            )
            futures.append(future)
            self.round_states[round_idx + 1][client.idx].update({"agg": future})
        self.aggregation_scheduler.step(round_idx)

        return futures  # results
