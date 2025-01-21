from __future__ import annotations

import sys
import time
from torch import nn
from torch.utils.data import Dataset
from datetime import datetime
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F  # noqa: N812

# TODO (MS): update all of these clients to decentralized clients
from src.decentralized_client import DecentralClient
from src.types import Result

from parsl.app.app import python_app


def accuracy(inputs, logits):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    # converts logits to predictions
    predictions = torch.argmax(shift_logits, axis=-1)

    # Now compute accuracy
    N = torch.numel(predictions)
    accuracy = (shift_labels == predictions).sum() / N

    return accuracy.cpu().item()


@python_app(executors=["decentral_train"])
def no_local_train(
    future: tuple(list[Result], DecentralClient),
    round_idx: int,
    epochs: int,
    batch_size: int,
    lr: float,
    prox_coeff: float,
    # device: torch.device,
    seed: int,
    backdoor: bool = False,
    dataset: DataChoices = None,
    *neighbor_futures: list[(list[Result], DecentralClient)],
) -> tuple(list[Result], DecentralClient):
    """Local training job.

    Args:
        client: The client to train.
        round_idx: The current round number.
        epochs: Number of epochs.
        batch_size: Batch size when iterating through data.
        lr: Learning rate.
        device: Backend hardware to train with.

    Returns:
        List of results that record the training history.
    """
    # from datetime import datetime
    # import torch
    # from torch.utils.data import DataLoader
    # from torch.nn import functional as F  # noqa: N812

    # NOTE(MS): assign device once task has been fired off, rather than before via a function arg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = dataset.value.lower()

    if seed is not None:
        torch.manual_seed(seed)

    client = future[1]
    results: list[Result] = []
    client.model.to(device)
    client.model.train()
    optimizer = torch.optim.SGD(client.model.parameters(), lr=lr)
    loader = DataLoader(client.train_data, batch_size=batch_size)

    for epoch in range(epochs):
        epoch_results = []
        n_batches = 1
        running_loss = 0.0

        # Test client on local test set TODO

        # Test client on global test set
        global_test_result = test_model(
            client.model,
            client.global_test_data,
            round_idx,
            batch_size,
            # device,
            seed,
            dataset,
        )

        epoch_results.append(
            {
                "time": datetime.now(),
                "client_idx": client.idx,
                "neighbors": client.neighbors,
                "round_idx": round_idx,
                "epoch": epoch,
                "data_size": len(client.train_data),
                "train_loss": running_loss / n_batches,
            }
            | global_test_result,
        )

        results.extend(epoch_results)

    return results, client


@python_app(executors=["decentral_train"])
def local_train(
    future: tuple(list[Result], DecentralClient),
    round_idx: int,
    epochs: int,
    batch_size: int,
    lr: float,
    prox_coeff: float,
    # device: torch.device,
    seed: int,
    backdoor: bool = False,
    dataset: DataChoices = None,
    *neighbor_futures: list[(list[Result], DecentralClient)],
) -> tuple(list[Result], DecentralClient):
    """Local training job.

    Args:
        client: The client to train.
        round_idx: The current round number.
        epochs: Number of epochs.
        batch_size: Batch size when iterating through data.
        lr: Learning rate.
        device: Backend hardware to train with.

    Returns:
        List of results that record the training history.
    """
    # NOTE(MS): assign device once task has been fired off, rather than before via a function arg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = dataset.value.lower()

    if seed is not None:
        print(f"{seed=}")
        torch.manual_seed(seed)

    client = future[1]
    results: list[Result] = []
    client.model.to(device)
    client.model.train()
    optimizer = torch.optim.SGD(client.model.parameters(), lr=lr)
    loader = DataLoader(client.train_data, batch_size=batch_size)

    avg_time_per_epoch = 0
    for epoch in range(epochs):
        start_time = time.time()

        epoch_results = []
        running_loss = 0.0
        running_perp = 0.0
        running_acc = 0.0

        for batch_idx, batch in enumerate(loader):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            if dataset_name == "tiny_mem":
                model_output = client.model(inputs, labels=inputs)
                loss = model_output.loss
                running_perp += torch.exp(loss).cpu().item()
                running_acc += accuracy(inputs, model_output.logits)

            else:
                preds = client.model(inputs)
                loss = F.cross_entropy(preds, targets)

            running_loss += loss.item()

            # Append proximal term
            # Inspired by: https://github.com/ki-ljl/FedProx-PyTorch/blob/main/client.py#L62
            if prox_coeff > 0:
                proximal_term = 0.0
                for neighbor_future in neighbor_futures:
                    neighbor = neighbor_future[1]
                    neighbor.model.to(device)
                    for w, w_t in zip(
                        client.model.parameters(), neighbor.model.parameters()
                    ):
                        proximal_term += (w - w_t).norm(2)
                loss += (prox_coeff / 2) * proximal_term

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        end_time = time.time()
        avg_time_per_epoch += end_time - start_time

        # Test client on local test set TODO

        # Test client on global test set
        global_test_result = test_model(
            client.model,
            client.global_test_data,
            round_idx,
            batch_size,
            seed,
            dataset,
        )

        if backdoor:
            # Test client on global backdoor test set
            global_backdoor_test_result = test_model(
                client.model,
                client.global_backdoor_test_data,
                round_idx,
                batch_size,
                seed,
            )

            global_test_result = global_test_result | {
                "backdoor_acc": global_backdoor_test_result["test_acc"],
                "backdoor_loss": global_backdoor_test_result["test_loss"],
            }
        epoch_results.append(
            {
                "avg_time_per_epoch": avg_time_per_epoch / epochs,
                # "avg_time_per_epoch": (start_time - end_time).total_seconds(),
                "date_time": datetime.now(),
                "client_idx": client.idx,
                "neighbors": client.neighbors,
                "round_idx": round_idx,
                "epoch": epoch,
                "data_size": len(client.train_data),
                "train_loss": running_loss / len(loader),
                "train_perp": running_perp / len(loader),
                "train_acc": running_acc / len(loader),
            }
            | global_test_result,
        )

        results.extend(epoch_results)

    # need to send model back to cpu since all coordination is happening from a non-gpu node
    client.model.to("cpu")
    return results, client


def test_model(
    model: nn.Module,
    data: Dataset,
    round_idx: int,
    batch_size: int,
    # device: torch.device,
    seed: int,
    dataset: DataChoices = None,
) -> Result:
    """Evaluate a model."""

    # NOTE(MS): assign device once task has been fired off, rather than before via a function arg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = dataset.value.lower()

    if seed is not None:
        torch.manual_seed(seed)

    if dataset_name == "tiny_mem":
        model.eval()
        total_loss, total_perp, total_acc = 0.0, 0.0, 0.0
        with torch.no_grad():
            model.to(device)
            loader = DataLoader(data, batch_size=batch_size)
            for batch in loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                model_outputs = model(inputs, labels=inputs)
                loss = model_outputs.loss
                total_loss += loss.item()
                total_perp += torch.exp(loss).cpu().item()
                total_acc += accuracy(inputs, model_outputs.logits)
        res: Result = {
            "test_loss": total_loss / len(loader),
            "test_perp": total_perp / len(loader),
            "test_acc": total_acc / len(loader),
        }
        return res

    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        model.to(device)
        loader = DataLoader(data, batch_size=batch_size)
        for batch in loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            y_true.extend(targets.cpu().tolist())
            loss = F.cross_entropy(preds, targets)
            total_loss += loss.item()

            # Accuracy calculations
            top_probabilities, top_preds = torch.topk(preds, k=1, dim=1)
            top_preds = torch.squeeze(top_preds)
            y_pred.extend(top_preds.cpu().tolist())
            acc = torch.sum(top_preds == targets) / batch_size
            total_acc += acc.item()

            n_batches += 1

    # print("test acc: ", total_acc / n_batches)
    report = classification_report(
        y_true, y_pred, digits=4, output_dict=True, zero_division=0
    )

    stats_dict = {}
    for k in report:
        if dict == type(report[k]):
            # print(k, report[k])
            for header in report[k]:
                stats_dict[f"{k}_{header}"] = report[k][header]
                # print(f'{k}_{header}: {report[k][header]}')
    # print(stats_dict)

    res: Result = {
        # "time": datetime.now(),
        # "round_idx": round_idx,
        "test_loss": total_loss / n_batches,
        "test_acc": total_acc / n_batches,
    } | stats_dict
    return res
