from concurrent.futures import as_completed
import pathlib
import torch
import pandas as pd
import logging
from typing import Optional
import os
import numpy as np

from src.types import Result
from src.decentralized_client import DecentralClient
from src.aggregation_scheduler import BaseScheduler

DEFAULT_FORMAT = (
    "DECENTRAL_TRAIN LOGGER: "
    "%(created)f %(asctime)s %(processName)s-%(process)d "
    "%(threadName)s-%(thread)d %(name)s:%(lineno)d %(funcName)s %(levelname)s: "
    "%(message)s"
)


def save_checkpoint(
    round_idx: int,
    clients: list[DecentralClient],
    client_results: list[Result],
    ckpt_path: pathlib.Path,
):
    client_state_dicts = []
    for client in clients:
        client_state_dicts.append(client.model.state_dict())

    ckpt = {
        "client_state_dicts": client_state_dicts,
        "round_idx": round_idx,
        "client_results": client_results,
    }

    torch.save(ckpt, ckpt_path)

    return


def load_checkpoint(
    ckpt_path: pathlib.Path,
    clients: list[DecentralClient],
    softmax_coeff_scheduler: BaseScheduler,
) -> tuple[int, list[DecentralClient], list[Result], BaseScheduler]:

    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    for i in range(len(clients)):
        sd = ckpt["client_state_dicts"][i]
        clients[i].model.load_state_dict(sd)

    # get softmax coeff scheduler to the correct point
    for i in range(ckpt["round_idx"]):
        softmax_coeff_scheduler.step(i)

    return ckpt["round_idx"], clients, ckpt["client_results"], softmax_coeff_scheduler


def process_futures_and_ckpt(
    client_results_init: list[Result],
    train_result_futures: tuple[list[Result], DecentralClient],
    round_states: dict[int, dict[int, tuple[list[Result], DecentralClient]]],
    rounds: int,
    run_dir: pathlib.Path,
) -> None:
    client_results = client_results_init.copy()
    # NOTE(MS): need to handle the case where rounds < ckpted rounds
    # aka user requested a shorter experiment than what exists
    # since we save a ckpt to round_idx - 1, make the same comparison here
    if rounds < (max(round_states.keys()) - 1):
        return

    ######### Process and Save training results
    resolved_futures = [i.result() for i in as_completed(train_result_futures)]
    [client_results.extend(i[0]) for i in resolved_futures]
    ckpt_clients = []
    for client_idx, client_future in round_states[rounds].items():
        result_object = client_future["agg"]
        # This is how we handle clients that are not returning appfutures (due to not being selected)
        if isinstance(result_object[1], DecentralClient):
            client = client_future["agg"][1]
        else:
            client = client_future["agg"].result()[1]
        ckpt_clients.append(client)
    # NOTE (MS): we only train until N-1 round so name ckpt accordingly
    checkpoint_path = f"{run_dir}/{rounds-1}_ckpt.pth"

    retries = 0
    max_retries = 3
    while retries < max_retries:
        retries += 1
        save_checkpoint(rounds - 1, ckpt_clients, client_results, checkpoint_path)

        if not os.path.exists(checkpoint_path):
            if retries >= max_retries:
                raise FileNotFoundError(
                    f"Error: The checkpoint '{checkpoint_path}' was not saved on disk."
                )
                return 2
        else:
            # ckpt is saved...break loop
            break

    client_df = pd.DataFrame(client_results)
    client_df.to_csv(f"{run_dir}/client_stats.csv")

    return 0


def set_file_logger(
    filename: str,
    name: str = "parsl",
    level: int = logging.DEBUG,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Add a file log handler.

    Args:
        - filename (string): Name of the file to write logs to
        - name (string): Logger name
        - level (logging.LEVEL): Set the logging level.
        - format_string (string): Set the format string

    Returns:
       - logger for specified name
    """
    if format_string is None:
        format_string = DEFAULT_FORMAT

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(level)
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # see note in set_stream_logger for notes about logging
    # concurrent.futures
    futures_logger = logging.getLogger("concurrent.futures")
    futures_logger.addHandler(handler)

    return logger


# utils for weighting aggregation
def constrained_sinkhorn(A, R, temperature=1.0, max_iter=10):
    """
    Creates a doubly stochastic matrix while preserving the sparsity
    structure of A_original and enforcing ||X - A|| <= R.
    """
    A_original = A.copy()
    # 1. Create a mask: 1 where A has an edge, 0 otherwise
    mask = (A_original != 0).astype(float)

    # 2. Initialize with the Gibbs kernel and apply mask immediately
    # We use a small epsilon to avoid log(0) issues if needed,
    # but since we multiply by mask, zeroed entries stay zero.
    X = np.exp(A_original / temperature) * mask

    for i in range(max_iter):
        X_old = X.copy()

        # --- Sinkhorn Steps (with Masking) ---
        # Row Normalization
        X /= X.sum(axis=1, keepdims=True) + 1e-9
        X *= mask  # Re-apply mask to ensure zeroed entries don't creep back

        # Column Normalization
        X /= X.sum(axis=0, keepdims=True) + 1e-9
        X *= mask

        # --- Similarity Constraint (Frobenius) ---
        diff = X - A_original
        dist = np.linalg.norm(diff, ord="fro")

        if dist > R:
            # Pull X back toward A_original
            X = A_original + (diff * (R / dist))
            # Ensure the projection respects the sparsity structure
            X *= mask

        # recommended tolerance is 1/(# nodes)
        tol = 1 / X.shape[0]
        # Convergence check
        if np.linalg.norm(X - X_old, ord="fro") < tol:
            break

    return X


def compute_mh_weights(adj_matrix):
    """
    Computes doubly stochastic weights (Metropolis-Hastings) for a graph.
    Assumes undirected graph (symmetric adjacency matrix).
    """
    # 1. Compute degrees for each node
    degrees = np.sum(adj_matrix, axis=1)
    n = adj_matrix.shape[0]
    W = np.zeros((n, n))

    # 2. Compute off-diagonal weights: W_ij = 1 / max(d_i, d_j)
    for i in range(n):
        for j in range(n):
            if i != j and adj_matrix[i, j] > 0:
                W[i, j] = 1.0 / max(degrees[i], degrees[j])

    # 3. Compute diagonal weights: W_ii = 1 - sum(W_ij, j!=i)
    for i in range(n):
        W[i, i] = 1.0 - np.sum(W[i, :])

    return W


def get_client_aggregation_weights(
    adj_mat: np.array,
    client_idx: int,
    neighbor_idxs: list[int],
):
    if not isinstance(adj_mat, np.ndarray):
        # adj_mat doesn't exist, fall back on different agg strategy
        return None

    weights = []
    for neighbor_idx in neighbor_idxs:
        weights.append(adj_mat[client_idx][neighbor_idx].item())
    return weights


def get_adj_mat(
    centrality_metric: str,
    softmax_coeff: float,
    softmax_bool: bool,
    matrix_type: str,
    topology: np.array,
    centrality_dict: dict[str, dict[int, float]],
    R: float = 3,
    sink_temp: float = 0.1,
):
    """
    centrality_metirc: name of centrality metric that informs weights
    softmax_coeff: how much to scale the softmax normalization step by
    softmax_bool: apply softmax...true or false
    matrix_type: n/a, sinkhorn_knopp (doubly-stochastic), frob_proj (doubly-stochastic)
    topology: np array encoding adj mat of array
    client_idx: client which is aggregating over its neighborhood
    neighbor_idxs: list of neighbors which are being aggregated over
    centrality_dict: a dict w/ all of the up-to-date centrality metrics for each node in topology
    R: radius for frobenius norm for double stoch procedure
    sink_temp: hp for double stoch procedure (doesn't make a huge difference)
    client: list of clients, needed for 'weighted' aggregation strategy
    """

    if centrality_metric not in [
        "degree",
        "betweenness",
        "random",
        "closeness",
        "eigen",
        "metro_hast",
    ]:
        return None

    adj_mat = topology.copy()
    adj_mat[np.diag_indices_from(adj_mat)] = 1

    """ NOTE(MS): we don't need unweighted/weighted rn
  
  if 'unweighted_fl' in centrality_metric:
    adj_mat = np.ones(shape=adj_mat.shape)

  if "unweighted" in centrality_metric:
    # unweighted/unweighted FL are already double stochastic
    # Reshape row_sums to a column vector to enable broadcasting
    # [:, None] adds a new axis, changing the shape from (3,) to (3, 1)
    row_sums = adj_mat.sum(axis=1)
    row_sums_reshaped = row_sums[:, np.newaxis]
    # Divide the original array by the column vector of row sums
    return = adj_mat / row_sums_reshaped
  """

    if centrality_metric == "metro_hast":
        return compute_mh_weights(adj_mat)
    # include self in the aggregation neighborhood for normalization purposes

    for row in range(adj_mat.shape[0]):

        # get all non-zero weights
        weights = [
            centrality_dict[centrality_metric][idx]
            for idx, i in enumerate(adj_mat[row])
            if i != 0
        ]

        if softmax_bool:

            def softmax(x):
                """Compute softmax values for each sets of scores in x."""
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()

            weights = [x * softmax_coeff for x in weights]
            weights = softmax(weights)

        mask = adj_mat[row] != 0
        adj_mat[row, mask] = weights
        zero_mask = adj_mat == 0

    softmax_weights = adj_mat.copy()

    if matrix_type == "sinkhorn_knopp":
        adj_mat = constrained_sinkhorn(adj_mat, R=R, temperature=sink_temp)

    return adj_mat
