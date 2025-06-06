from __future__ import annotations

from collections import OrderedDict
from typing import Optional
from typing import OrderedDict

import sys
import torch
import json
import numpy as np
from numpy.random import Generator
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
import torch.nn as nn
import networkx as nx

from src.modules import create_model
from src.types import DataChoices
from src.data import federated_split
from src.data import random_generator
from src.data import backdoor_data

from parsl.app.app import python_app


class DecentralClient(BaseModel):
    """Client class."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    idx: int = Field(description="Client ID.")
    prox_coeff: float = Field(description="Proximal term coefficient (FedProx).")
    model: torch.nn.Module = Field(description="Client local model.")
    train_data: Optional[Subset] = Field(  # noqa: UP007
        description="Subset of data this client will train on.",
    )
    test_data: Optional[Subset] = Field(  # noqa: UP007
        description="Subset of local data this client will test on.",
    )
    valid_data: Optional[Subset] = Field(  # noqa: UP007
        description="Subset of local data this client will validate on.",
    )
    global_test_data: Dataset = Field(
        description="global set of test data that every client and global model is evaluated on."
    )
    global_backdoor_test_data: Optional[Subset] = Field(  # noqa: UP007
        description="Subset of global test data that has been backdoored (for testing ASR).",
    )
    neighbors: list[int] = Field(description="list of this clients neighbors")
    neighbor_probs: list[float] = Field(
        description="list of this clients neighbors' network connection probabilities (for modeling faulty networks)"
    )
    # NOTE(MS): currenlty we maintain a glocal view of the centrlaity metrics in the main app,
    # NOTE (MS): maybe in the future it may make sense for each node to have a local view of the topology centrality measures
    # centrality_dict: dict[str, dict[int, float]] = Field(
    #    description="Dict to track node-wise centrality metrics"
    # )

    def get_neighbors(self) -> list[ints]:
        neighbor_idxs = self.neighbors
        neighbor_probs = self.neighbor_probs
        # This is where we set the probability of including a speicfic neighbor in that aggregation round
        # simulating faulty networks
        prob_idxs = np.random.binomial(1, neighbor_probs)
        # mask out any neighbors that don't make the inclusion threshold
        neighbor_idxs = [a for a, b in zip(neighbor_idxs, prob_idxs) if b > 0]
        return neighbor_idxs


def get_label_counts(
    num_clients: int,
    num_labels: int,
    subsets: dict[int, Subset],
) -> dict[
    int, list[int]
]:  # return dict of lists (each key is a label), list is the length of # of clients
    """return label count dict for each data split"""

    # save label counts per worker
    label_counts_per_worker = {label: [0] * num_clients for label in range(num_labels)}

    for idx in range(num_clients):
        for batch in subsets[idx]:
            _, label = batch
            label_counts_per_worker[label][idx] += 1

    return label_counts_per_worker


def place_data_with_node(
    label_counts_per_worker: dict[int, list[int]],
    centrality_dict: dict[str, dict[int, float]],
    data: Dataset,
    train_indices: dict[int, list[int]],
    test_indices: dict[int, list[int]],
    valid_indices: dict[int, list[int]],
    num_clients: int,
    offset_clients_data_placement: int = 0,  # this is how many clients we off set the data assignment by
    centrality_metric_data_placement: str = "degree",
    random_data_placement: bool = True,
) -> dict[int, Subset]:
    """Function used to place data with client"""
    test_subsets = {idx: None for idx in range(num_clients)}
    valid_subsets = {idx: None for idx in range(num_clients)}
    if random_data_placement:
        print("RANDOM DATA PLACEMENT")
        train_subsets = {
            idx: Subset(data, train_indices[idx]) for idx in range(num_clients)
        }
        if test_indices is not None:
            test_subsets = {idx: Subset(data, test_indices[idx]) for idx in client_ids}
        if valid_indices is not None:
            valid_subsets = {
                idx: Subset(data, valid_indices[idx]) for idx in client_ids
            }

    else:
        # data placement based on centrality metric
        print(
            f"DATA PLACEMENT w/ {centrality_metric_data_placement=}, {offset_clients_data_placement=}"
        )
        centrality_list = [
            x for k, x in centrality_dict[centrality_metric_data_placement].items()
        ]
        sorted_nodes = [
            x for _, x in sorted(zip(centrality_list, list(range(num_clients))))
        ]
        print(f"{sorted_nodes=}")

        # TODO(MS): in the future sort by something in addition to # of samples
        data_len_list = [len(x) for _, x in train_indices.items()]
        sorted_data = [
            x for _, x in sorted(zip(data_len_list, list(range(num_clients))))
        ]
        print(f"{sorted_data=}")
        for i in range(offset_clients_data_placement):
            temp = sorted_data.pop(0)
            sorted_data.append(temp)

        train_subsets = {
            sorted_nodes[idx]: Subset(data, train_indices[sorted_data[idx]])
            for idx in range(num_clients)
        }
        if test_indices is not None:
            test_subsets = {
                sorted_nodes[idx]: Subset(data, test_indices[sorted_data[idx]])
                for idx in range(num_clients)
            }
        if valid_indices is not None:
            valid_subsets = {
                sorted_nodes[idx]: Subset(data, valid_indices[sorted_data[idx]])
                for idx in range(num_clients)
            }
    return train_subsets, test_subsets, valid_subsets


def update_random_agg_coeffs(
    seed: int,
    round_idx: int,
    num_clients: int,
    centrality_dict: dict[str, dict[int, float]],
) -> dict[str, dict[int, float]]:
    # generate random coeffs
    # seed based on current round
    rng = np.random.default_rng(seed=(seed + round_idx))
    random_coeffs = rng.uniform(low=0.0, high=1.0, size=num_clients)

    # format them in a client_idx:coeff dict
    cent = {}
    for i in range(num_clients):
        cent[i] = random_coeffs[i].item()

    print(f"Random Coeffs: {cent=}")
    # assign updated random dict
    centrality_dict["random"] = cent

    return centrality_dict


def create_centrality_dict(
    topology: np.array,  # list[list[int]],
    rng: Generator,
) -> dict[str, dict[int, float]]:
    print("Creating centrality dict")
    # convert np array to graph
    G = nx.from_numpy_array(topology)

    centrality_dict = {}
    for centrality_type in ["degree", "betweenness", "random"]:
        if centrality_type == "degree":
            cent = nx.degree_centrality(G)
        if centrality_type == "betweenness":
            cent = nx.betweenness_centrality(G, normalized=True, endpoints=True)
        """
        if centrality_type == "cluster":
            cent = nx.degree_centrality(G)
        """
        if centrality_type == "random":
            # generator = random_generator(rng)
            # random_list = generator.dirichlet(np.ones(len(G)))
            # random_list = np.random.dirichlet(np.ones(len(G)), size=1).squeeze()
            # random_list =
            random_coeffs = rng.uniform(low=0.0, high=1.0, size=len(G))
            cent = {}
            for i in range(len(G)):
                cent[i] = random_coeffs[i].item()
            print(f"{cent=}")
        """
        if centrality_type == "invCluster":
            cent = nx.degree_centrality(G)
            for k, v in cent.items():
                cent[k] = 1 / v
        """
        centrality_dict[centrality_type] = cent

    # This function returns a dict of different types of centrality dicts
    return centrality_dict


def create_clients(
    num_clients: int,
    data_name: DataChoices,
    # train: bool,
    train_data: Dataset,
    num_labels: int,
    global_test_data: Dataset,
    label_alpha: float,
    sample_alpha: float,
    rng: Generator,
    topology: np.array,  # list[list[int]],
    prox_coeff: float,
    run_dir: pathlib.Path,
    train_test_val_split: tuple[float],
    backdoor_test_data: Dataset,
    backdoor: bool,
    backdoor_proportion: float,
    backdoor_node_idx: int,
    random_bd: bool = False,
    # many-to-many or many-to-one backdoor from https://arxiv.org/pdf/1708.06733
    many_to_one: bool = True,
    offset_clients_data_placement: int = 0,  # this is how many clients we off set the data assignment by
    centrality_metric_data_placement: str = "degree",
    random_data_placement: bool = True,
    ckpt_dir: str = "./ckpt",
    trigger: int = 100,
) -> list[DecentralClient]:
    """Create many clients with disjoint sets of data.

    Args:
        num_clients: Number of clients to create.
        data_name: The name of the data used. Used for initializing the
            corresponding model.
        train: If the application is using the no-op training task, then this
            function skips the step for giving each client their own subset
            of data.
        train_data: The original dataset that will be split across the clients.
        global_test_data: The global test dataset that each client will be assessed on.
        data_alpha: The
            [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution)
            distribution alpha value for the number of samples across clients.
        rng: Random number generator.

    Returns:
        List of clients.
    """
    client_ids = list(range(num_clients))

    client_indices: dict[int, list[int]] = {idx: [] for idx in client_ids}

    # TODO(MS) to create balanced local train and test sets
    # TODO(MS): pass in train_test_valid_split
    (train_indices, test_indices, valid_indices) = federated_split(
        ckpt_dir=ckpt_dir,
        # data_name=data_name,
        num_workers=num_clients,
        data=train_data,
        num_labels=num_labels,
        label_alpha=label_alpha,
        sample_alpha=sample_alpha,
        # train_test_valid_split=None,
        train_test_valid_split=train_test_val_split,
        # train_test_valid_split=(0.7, 0.2, 0.1),
        ensure_at_least_one_sample=True,
        rng=rng,
        allow_overlapping_samples=False,
    )
    # print(f"{len(train_indices[0])=}")

    centrality_dict = create_centrality_dict(topology, rng=rng)
    train_subsets = {idx: Subset(train_data, train_indices[idx]) for idx in client_ids}
    label_counts_per_worker = get_label_counts(
        num_clients=len(client_ids),
        num_labels=num_labels,
        subsets=train_subsets,
    )

    train_subsets, test_subsets, valid_subsets = place_data_with_node(
        label_counts_per_worker,
        centrality_dict,
        data=train_data,
        train_indices=train_indices,
        test_indices=test_indices,
        valid_indices=valid_indices,
        num_clients=len(client_ids),
        offset_clients_data_placement=offset_clients_data_placement,  # experiment arg
        centrality_metric_data_placement=centrality_metric_data_placement,  # experiment arg
        random_data_placement=random_data_placement,  # experiment arg (MS): default behavior needs to be over turned in the args when making a client
    )

    if backdoor:
        rng_seed = rng.integers(low=0, high=4294967295, size=1).item()
        stratify_targets = [label for x, label in train_subsets[backdoor_node_idx]]
        clean_data, bd_data = backdoor_data(
            data_name.value.lower(),
            train_subsets[backdoor_node_idx],
            stratify_targets,
            backdoor_proportion,
            rng_seed,
            rng,
            num_labels,
            random_bd,
            many_to_one,
            # for propoer checkpointing purposes we need to save some additional info
            offset_clients_data_placement,
            centrality_metric_data_placement,
            random_data_placement,
            backdoor_node_idx,
            num_clients=len(client_ids),
            test_data=0,  # this is trianing data
            trigger=trigger,
        )
        # combine clean + bd training data
        concat_data = ConcatDataset([clean_data, bd_data])
        clean_len = len(clean_data)
        bd_len = len(bd_data)
        new_indices = list(range(clean_len + bd_len))
        # new_indices = list(range(len(stratify_targets)))
        # wrap new bd-ed data in Subset class
        train_subsets[backdoor_node_idx] = Subset(concat_data, new_indices)
        print(f"backdoored client {backdoor_node_idx} data")

    clients = []
    for idx in client_ids:
        neighbors = np.where(topology[idx] > 0)[0].tolist()
        prob_idxs = np.argwhere(topology[idx] > 0)
        probs = (topology[idx][prob_idxs]).flatten().tolist()
        print(f"client: {idx}, neighbors: {neighbors}, neighbor probabilities: {probs}")
        # print(f"train data size: {len(train_subsets[idx])}")
        client = DecentralClient(
            idx=idx,
            model=create_model(data_name),
            train_data=train_subsets[idx],
            test_data=test_subsets[idx],
            valid_data=valid_subsets[idx],
            global_test_data=global_test_data,
            neighbors=neighbors,
            neighbor_probs=probs,
            prox_coeff=prox_coeff,
            global_backdoor_test_data=backdoor_test_data,
            # centrality_dict=centrality_dict,
        )
        clients.append(client)

    label_counts_per_worker = get_label_counts(
        num_clients=len(client_ids),
        num_labels=num_labels,
        # indices=train_indices,
        # data=train_data,
        subsets=train_subsets,
    )

    json.dump(
        label_counts_per_worker, open(f"{run_dir}/label_counts_per_worker.txt", "w")
    )

    return clients


@python_app(executors=["threadpool_executor"])
def weighted_module_avg(
    client_future: tuple(list[Result], DecentralClient),
    seed: int,
    *neighbor_futures: list[(list[Result], DecentralClient)],
    **kwargs: str,  # placeholder for keyword args
) -> tuple(list[Result], DecentralClient):
    """Compute the weighted average of models."""
    # import torch

    if seed is not None:
        torch.manual_seed(seed)
    print("weighted aggregate round")
    data_lens = [len(client_future[1].train_data) for client_future in neighbor_futures]
    weights = [x / sum(data_lens) for x in data_lens]

    with torch.no_grad():
        avg_weights = OrderedDict()
        for i in range(len(neighbor_futures)):
            client = neighbor_futures[i]
            model = client[1].model
            model.to("cpu")
            w = weights[i]
            for name, value in model.state_dict().items():
                partial = w * torch.clone(value)
                if name not in avg_weights:
                    avg_weights[name] = partial
                else:
                    avg_weights[name] += partial

    client_future[1].model.load_state_dict(avg_weights)
    return client_future
    # return (client_future[0], client_future[1])


@python_app(executors=["threadpool_executor"])
def unweighted_module_avg(
    client_future: tuple(list[Result], DecentralClient),
    seed: int,
    *neighbor_futures: list[(list[Result], DecentralClient)],
    **kwargs: str,  # placeholder for keyword args
) -> tuple(list[Result], DecentralClient):
    """Compute the unweighted average of models."""
    # import torch

    if seed is not None:
        torch.manual_seed(seed)
    print("unweighted aggregate round")
    w = 1 / len(neighbor_futures)

    with torch.no_grad():
        avg_weights = OrderedDict()
        for client in neighbor_futures:
            model = client[1].model
            model.to("cpu")
            # model = client.result()[1].model
            for name, value in model.state_dict().items():
                partial = w * torch.clone(value)
                if name not in avg_weights:
                    avg_weights[name] = partial
                else:
                    avg_weights[name] += partial

    client_future[1].model.load_state_dict(avg_weights)

    return client_future


@python_app(executors=["threadpool_executor"])
def sim_centrality_module_avg(
    client_future: tuple(list[Result], DecentralClient),
    seed: int,
    *neighbor_futures: list[(list[Result], DecentralClient)],
    **kwargs: str,  # type of centrality metric
) -> tuple(list[Result], DecentralClient):
    """Compute the weighted average of models."""
    # import torch
    if seed is not None:
        torch.manual_seed(seed)

    # Grab centrality dict, centrality metric
    centrality_dict = kwargs["centrality_dict"]
    centrality_metric = kwargs["centrality_metric"]
    softmax = kwargs["softmax"]
    softmax_coeff = kwargs["softmax_coeff"]
    print(f"{centrality_metric} aggregate round w/ {softmax=}")

    # get weights
    weights = []
    neighborhood_weights = {}
    for n in neighbor_futures:
        idx = n[1].idx
        weight = centrality_dict[centrality_metric][idx]
        weights.append(weight)
        neighborhood_weights[idx] = weight

    # calculate similarity between client and its neighbor
    neighborhood_sim = {}
    client_model = client_future[1].model
    for i in range(len(neighbor_futures)):
        client = neighbor_futures[i]
        neighbor_model = client[1].model
        neighbor_idx = client[1].idx
        if neighbor_idx == client_future[1].idx:
            # these are both the aggregating client: same model
            continue
        neighbor_model.to("cpu")
        neighborhood_sim[neighbor_idx] = cosine_similarity(client_model, neighbor_model)

    # convert the softmax to positive or negative
    client_idx = client_future[1].idx
    client_weight = neighborhood_weights[client_idx]
    min_weights = min(neighborhood_weights, key=neighborhood_weights.get)
    max_weights = max(neighborhood_weights, key=neighborhood_weights.get)

    # NOTE(MS): the two explicit conditions below are handed by the 3rd setting
    """
    # lowest degree client in neighborhood, consume neighbors
    if client_idx == min_weights:
        softmax_coeff = abs(softmax_coeff) 

    # highest degree client in neighborhood, consume neighbors
    if client_idx == min_weights:
        softmax_coeff = -1 * abs(softmax_coeff) 
    """

    # middle degree client in neighborhood,
    # consume neighbors with highest dissimilarity
    min_similarity = min(neighborhood_sim, key=neighborhood_sim.get)
    deg_for_dissimilar_node = neighborhood_weights[min_similarity]
    if deg_for_dissimilar_node < client_weight:
        softmax_coeff = -abs(softmax_coeff)
    else:
        softmax_coeff = abs(softmax_coeff)

    # scale the weights by softmax_coeff
    if softmax:
        print(f"{client_idx=} softmaxing aggregation weights w/ {softmax_coeff=}")

        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        weights = [x * softmax_coeff for x in weights]
        weights = softmax(weights)

    else:
        print("1/N aggregation weights")
        # normalize weights
        weights = [i / sum(weights) for i in weights]

    with torch.no_grad():
        avg_weights = OrderedDict()
        for i in range(len(neighbor_futures)):
            client = neighbor_futures[i]
            model = client[1].model
            model.to("cpu")
            w = weights[i]
            for name, value in model.state_dict().items():
                partial = w * torch.clone(value)
                if name not in avg_weights:
                    avg_weights[name] = partial
                else:
                    avg_weights[name] += partial

    client_future[1].model.load_state_dict(avg_weights)
    return client_future


@python_app(executors=["threadpool_executor"])
def centrality_module_avg(
    client_future: tuple(list[Result], DecentralClient),
    seed: int,
    *neighbor_futures: list[(list[Result], DecentralClient)],
    **kwargs: str,  # type of centrality metric
) -> tuple(list[Result], DecentralClient):
    """Compute the weighted average of models."""
    # import torch
    if seed is not None:
        torch.manual_seed(seed)

    # Grab centrality dict, centrality metric
    centrality_dict = kwargs["centrality_dict"]
    centrality_metric = kwargs["centrality_metric"]
    softmax = kwargs["softmax"]
    softmax_coeff = kwargs["softmax_coeff"]
    print(f"{centrality_metric} aggregate round w/ {softmax=}")

    # get weights
    weights = []
    for n in neighbor_futures:
        idx = n[1].idx
        weights.append(centrality_dict[centrality_metric][idx])

    # print(f"pre-normalization {weights=}")
    if softmax:
        print(f"softmaxing aggregation weights w/ {softmax_coeff=}")

        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        weights = [x * softmax_coeff for x in weights]
        weights = softmax(weights)

    else:
        print("1/N aggregation weights")
        # normalize weights
        weights = [i / sum(weights) for i in weights]

    # print(f"{weights=}")

    with torch.no_grad():
        avg_weights = OrderedDict()
        for i in range(len(neighbor_futures)):
            client = neighbor_futures[i]
            model = client[1].model
            model.to("cpu")
            w = weights[i]
            for name, value in model.state_dict().items():
                partial = w * torch.clone(value)
                if name not in avg_weights:
                    avg_weights[name] = partial
                else:
                    avg_weights[name] += partial

    client_future[1].model.load_state_dict(avg_weights)
    return client_future


@python_app(executors=["threadpool_executor"])
def scale_agg(
    client_future: tuple(list[Result], DecentralClient),
    seed: int,
    *neighbor_futures: list[(list[Result], DecentralClient)],
    **kwargs: str,  # placeholder for keyword args
) -> tuple(list[Result], DecentralClient):
    """Compute the unweighted average of models."""
    # import torch

    if seed is not None:
        torch.manual_seed(seed)
    print("unweighted aggregate round")
    w = 1 / len(neighbor_futures)

    with torch.no_grad():
        avg_weights = OrderedDict()
        for client in [
            client_future,
        ]:
            model = client[1].model
            model.to("cpu")
            # model = client.result()[1].model
            for name, value in model.state_dict().items():
                partial = w * torch.clone(value)
                if name not in avg_weights:
                    avg_weights[name] = partial
                else:
                    avg_weights[name] += partial

    client_future[1].model.load_state_dict(avg_weights)

    return client_future


@python_app(executors=["threadpool_executor"])
def test_agg(
    client_future: tuple(list[Result], DecentralClient),
    seed: int,
    *neighbor_futures: list[(list[Result], DecentralClient)],
    **kwargs: str,  # placeholder for keyword args
) -> tuple(list[Result], DecentralClient):

    return client_future


# similarity metrics
def cosine_similarity(model_1, model_2):
    avg_cos = 0
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    weights_1 = []
    weights_2 = []
    # iterate over model layers
    with torch.no_grad():
        for model, weights in zip([model_1, model_2], [weights_1, weights_2]):
            for name, W in model.named_parameters():
                weights.append(W)

        # now do avg cosine sim
        for w_1, w_2 in zip(weights_1, weights_2):
            if len(w_1.shape) < 2:
                w_1 = w_1.unsqueeze(dim=1)
                w_2 = w_2.unsqueeze(dim=1)
            sim = cos(w_1, w_2)
            avg_cos += sim.mean()

    return avg_cos / len(weights_1)
