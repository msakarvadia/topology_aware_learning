from __future__ import annotations

from collections import OrderedDict
from typing import Optional

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

from src.modules import create_model
from src.types import DataChoices
from src.data import federated_split

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
    neighbors: list[int] = Field(description="list of this clients neighbors")
    neighbor_probs: list[float] = Field(
        description="list of this clients neighbors' network connection probabilities (for modeling faulty networks)"
    )
    # local_test_data: Dataset = Field(description="local test data that this client evaluated on.")

    def get_neighbors(self) -> list[ints]:
        neighbor_idxs = self.neighbors
        neighbor_probs = self.neighbor_probs
        # This is where we set the probability of including a speicfic neighbor in that aggregation round
        # simulating faulty networks
        prob_idxs = np.random.binomial(1, neighbor_probs)
        # mask out any neighbors that don't make the inclusion threshold
        neighbor_idxs = [a for a, b in zip(neighbor_idxs, prob_idxs) if b > 0]
        return neighbor_idxs


def create_clients(
    num_clients: int,
    data_name: DataChoices,
    train: bool,
    train_data: Dataset,
    num_labels: int,
    global_test_data: Dataset,
    label_alpha: float,
    sample_alpha: float,
    rng: Generator,
    topology: np.array,  # list[list[int]],
    prox_coeff: float,
    run_dir: pathlib.Path,
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

    if train:
        client_indices: dict[int, list[int]] = {idx: [] for idx in client_ids}

        # alpha = [sample_alpha] * num_clients
        # client_popularity = rng.dirichlet(alpha)

        # for data_idx, _ in enumerate(train_data):
        #    client_id = rng.choice(client_ids, size=1, p=client_popularity)[0]
        #    client_indices[client_id].append(data_idx)

        # TODO(MS) to create balanced local train and test sets
        # TODO(MS): pass in train_test_valid_split
        (train_indices, test_indices, valid_indices) = federated_split(
            num_workers=num_clients,
            data=train_data,
            num_labels=num_labels,
            label_alpha=label_alpha,
            sample_alpha=sample_alpha,
            train_test_valid_split=None,
            # train_test_valid_split=(0.7, 0.2, 0.1),
            ensure_at_least_one_sample=True,
            rng=rng,
            allow_overlapping_samples=False,
        )

        train_subsets = {
            idx: Subset(train_data, train_indices[idx]) for idx in client_ids
        }

        test_subsets = valid_subsets = None
        test_subsets = {idx: None for idx in client_ids}
        valid_subsets = {idx: None for idx in client_ids}
        if test_indices is not None:
            test_subsets = {
                idx: Subset(train_data, test_indices[idx]) for idx in client_ids
            }
        if valid_indices is not None:
            valid_subsets = {
                idx: Subset(train_data, valid_indices[idx]) for idx in client_ids
            }
    else:
        train_subsets = {idx: None for idx in client_ids}
        test_subsets = {idx: None for idx in client_ids}
        valid_subsets = {idx: None for idx in client_ids}

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
        )
        clients.append(client)

    # save label counts per worker
    label_counts_per_worker = {
        label: [0] * len(client_ids) for label in range(num_labels)
    }

    for idx in client_ids:
        for batch in train_subsets[idx]:
            _, label = batch
            label_counts_per_worker[label][idx] += 1

    json.dump(
        label_counts_per_worker, open(f"{run_dir}/label_counts_per_worker.txt", "w")
    )

    return clients


@python_app(executors=["threadpool_executor"])
def weighted_module_avg(
    neighbor_futures: list[tuple(list[Result], DecentralClient)],
    client_future: tuple(list[Result], DecentralClient),
    # selected_clients: list[DecentralClient],
) -> tuple(list[Result], DecentralClient):
    """Compute the weighted average of models."""
    models = [client_future[1].model for client_future in neighbor_futures]
    data_lens = [len(client_future[1].train_data) for client_future in neighbor_futures]
    weights = [x / sum(data_lens) for x in data_lens]

    with torch.no_grad():
        avg_weights = OrderedDict()
        for model, w in zip(models, weights):
            for name, value in model.state_dict().items():
                partial = w * torch.clone(value)
                if name not in avg_weights:
                    avg_weights[name] = partial
                else:
                    avg_weights[name] += partial

    client_future[1].model.load_state_dict(avg_weights)
    return (client_future[0], client_future[1])


@python_app(executors=["threadpool_executor"])
def unweighted_module_avg(
    neighbor_futures: list[tuple(list[Result], DecentralClient)],
    client_future: tuple(list[Result], DecentralClient),
    # selected_clients: list[DecentralClient],
) -> tuple(list[Result], DecentralClient):
    """Compute the unweighted average of models."""
    print(f"{neighbor_futures=}", file=sys.stderr)
    models = [client_future[1].model for client_future in neighbor_futures]
    print(f"aggregate {models=}", file=sys.stderr)
    w = 1 / len(models)

    with torch.no_grad():
        avg_weights = OrderedDict()
        for model in models:
            for name, value in model.state_dict().items():
                partial = w * torch.clone(value)
                if name not in avg_weights:
                    avg_weights[name] = partial
                else:
                    avg_weights[name] += partial

    client_future[1].model.load_state_dict(avg_weights)
    return (client_future[0], client_future[1])
