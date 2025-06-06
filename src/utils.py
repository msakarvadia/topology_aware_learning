from concurrent.futures import as_completed
import pathlib
import torch
import pandas as pd
import logging
from typing import Optional

from src.types import Result
from src.decentralized_client import DecentralClient
from src.aggregation_scheduler import BaseScheduler

DEFAULT_FORMAT = (
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
    print(f"Saved checkpoint for round: {round_idx}")

    return


def load_checkpoint(
    ckpt_path: pathlib.Path,
    clients: list[DecentralClient],
    softmax_coeff_scheduler: BaseScheduler,
) -> tuple[int, list[DecentralClient], list[Result], BaseScheduler]:

    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    for i in range(len(clients)):
        sd = ckpt["client_state_dicts"][i]
        clients[i].model.load_state_dict(sd)

    # get softmax coeff scheduler to the correct point
    for i in range(ckpt["round_idx"]):
        softmax_coeff_scheduler.step(i)

    return ckpt["round_idx"], clients, ckpt["client_results"], softmax_coeff_scheduler


def process_futures_and_ckpt(
    client_results: list[Result],
    train_result_futures: tuple[list[Result], DecentralClient],
    round_states: dict[int, dict[int, tuple[list[Result], DecentralClient]]],
    rounds: int,
    run_dir: pathlib.Path,
) -> None:

    # NOTE(MS): need to handle the case where rounds < ckpted rounds
    # aka user requested a shorter experiment than what exists
    # since we save a ckpt to round_idx - 1, make the same comparison here
    if rounds < (max(round_states.keys()) - 1):
        with open("out.txt", "a") as f:
            print(f"{rounds} < {max(round_states.keys())}", file=f)

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
    save_checkpoint(rounds - 1, ckpt_clients, client_results, checkpoint_path)

    client_df = pd.DataFrame(client_results)
    client_df.to_csv(f"{run_dir}/client_stats.csv")

    return


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
