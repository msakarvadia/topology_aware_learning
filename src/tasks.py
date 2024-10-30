from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.metrics import classification_report

# TODO (MS): update all of these clients to decentralized clients
from src.decentralized_client import DecentralClient
from src.types import Result

from parsl.app.app import python_app


def no_local_train(
    client: DecentralClient,
    round_idx: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> list[Result]:
    """No-op version of [local_train]

    Returns:
        Empty result list.
    """
    return []


def local_train(
    client: DecentralClient,
    round_idx: int,
    epochs: int,
    batch_size: int,
    lr: float,
    prox_coeff: float,
    device: torch.device,
    neighbors: list[DecentralClient],
) -> list[Result]:
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
    from datetime import datetime

    results: list[Result] = []
    client.model.to(device)
    client.model.train()
    optimizer = torch.optim.SGD(client.model.parameters(), lr=lr)
    loader = DataLoader(client.train_data, batch_size=batch_size)

    for epoch in range(epochs):
        epoch_results = []
        n_batches = 0
        # log_every_n_batches = 100
        running_loss = 0.0

        for batch_idx, batch in enumerate(loader):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            preds = client.model(inputs)
            loss = F.cross_entropy(preds, targets)

            # Append proximal term
            # Inspired by: https://github.com/ki-ljl/FedProx-PyTorch/blob/main/client.py#L62
            if prox_coeff > 0:
                proximal_term = 0.0
                for neighbor in neighbors:
                    neighbor.model.to(device)
                    for w, w_t in zip(
                        client.model.parameters(), neighbor.model.parameters()
                    ):
                        proximal_term += (w - w_t).norm(2)
                loss += (prox_coeff / 2) * proximal_term

            loss.backward()
            # print(client.model)
            # print(f"{loss=}, {torch.count_nonzero(client.model.fc1.weight.grad)=}")
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            n_batches += 1

        # Test client on local test set TODO

        # Test client on global test set
        global_test_result = test_model(
            client.model, client.global_test_data, round_idx, batch_size, device
        )

        epoch_results.append(
            {
                "time": datetime.now(),
                "client_idx": client.idx,
                "neighbors": client.neighbors,
                "round_idx": round_idx,
                "epoch": epoch,
                # "batch_idx": batch_idx,
                "data_size": len(client.train_data),
                "train_loss": running_loss / n_batches,
                # "global_test_acc": global_test_result["test_acc"],
                # "global_test_loss": global_test_result["test_loss"],
            }
            | global_test_result,
        )

        results.extend(epoch_results)

    return results


def test_model(
    model: nn.Module,
    data: Dataset,
    round_idx: int,
    batch_size: int,
    device: torch.device,
) -> Result:
    """Evaluate a model."""
    from datetime import datetime

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
