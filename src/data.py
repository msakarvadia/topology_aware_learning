from torch.utils.data import DataLoader
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import pathlib
import numpy as np
import math
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch
import os

from src.modules import load_data
from src.modules import CustomLMDataset
from src.types import DataChoices

import typing as t

if t.TYPE_CHECKING:
    from .types import T

T = t.TypeVar("T")
"""
Generic type variable used throughout Flight.
"""
FloatTriple: t.TypeAlias = tuple[float, float, float]


def proportion_split(
    seq: t.Sequence[T],
    proportion: t.Sequence[float],
    rng_seed: int | None = None,
    labels: t.Sequence[T] = None,
) -> tuple[t.Sequence[T], ...]:
    """
    Split a sequence into multiple sequences based on proportions.

    Args:
        seq (Sequence[T]): Sequence to split.
        proportion (t.Sequence[float, ...]): Proportions to split the sequence.

    Returns:
        Sequences split based on the proportions. The number of sequences returned is
            equal to the length of the `proportion` argument.

    Examples:
        >>> lst = list(range(10))
        >>> lst
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> proportion_split(lst, (0.5, 0.5))
        ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])
        >>> proportion_split(lst, (0.5, 0.2, 0.3))
        ([0, 1, 2, 3, 4], [5, 6], [7, 8, 9])

    Throws:
        - `ValueError`: If the number of proportions is greater than the number of
            values in the sequence.
        - `ValueError`: If the values in `proportion` argument are negative.
        - `ValueError`: If the values in `proportion` argument do not sum to 1.
    """
    if len(proportion) > 3:
        raise ValueError("Cannot make more than 3 splits (trian, test, val) ")
    if len(proportion) > len(seq):
        raise ValueError(
            "Number of proportions cannot be greater than the number of values in "
            "the sequence."
        )
    if any(p < 0 for p in proportion):
        raise ValueError("Proportions must be non-negative.")
    # if sum(proportion) != 1:
    # check if sum to 1, within a small tolerance for float arithmetic
    if not math.isclose(sum(proportion), 1):
        # print(sum(proportion))
        raise ValueError("Proportions must sum to 1.")

    # Need to ensure that test_size > # of classes
    num_classes = len(list(set(labels)))  # of classes

    test_size = int(len(seq) * proportion[-1])
    # print(f"{test_size=}")
    if test_size < num_classes:
        test_size = num_classes
    train, test, train_labels, test_labels = train_test_split(
        seq, labels, test_size=test_size, random_state=rng_seed, stratify=labels
    )
    if len(proportion) == 2:
        return (train, test)

    if len(proportion) == 3:
        # size of validation split
        test_size = int(len(train) * (proportion[1] / (proportion[0] + proportion[1])))
        if test_size < num_classes:
            test_size = num_classes
        train, val = train_test_split(
            train, test_size=test_size, random_state=rng_seed, stratify=train_labels
        )
        return (train, test, val)


def random_generator(
    rng: np.random.Generator | int | None = None,
) -> np.random.Generator:
    """
    Create a random number generator.

    Args:
        rng (numpy.random.Generator | int | None): Random number generator.

    Returns:
        Random number generator.

    Notes:
        What is returned by this function depends on what is given to the `rng` arg:

        1. If `rng` is a `numpy.random.Generator`, it is returned as is.
        2. If `rng` is an integer, it is used to seed the random number generator.
        3. If `rng` is `None`, then a pseudorandom  random number generator is
            returned using `numpy.random.default_rng(None)`.

    Throws:
        - `ValueError`: Is thrown if an illegal value type is passed in as an argument.
    """
    match rng:
        case np.random.Generator():
            return rng
        case int() | None:
            return np.random.default_rng(rng)
        case _:
            raise ValueError(
                f"Illegal value type for arg `rng`; expected a "
                f"`numpy.random.Generator`, int, or `None`, got "
                f"{type(rng)}."
            )


def federated_split(
    # data_name: str,
    ckpt_dir: str,
    num_workers: int,
    data: Dataset,
    num_labels: int,
    label_alpha: float,
    sample_alpha: float,
    train_test_valid_split: FloatTriple | None = None,
    ensure_at_least_one_sample: bool = True,
    rng: np.random.Generator | int | None = None,
    allow_overlapping_samples: bool = False,
) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, list[int]]]:
    """
    Splits a dataset across a federation of workers.

    The splitting of a dataset can be tuned via the `label_alpha` and `sample_alpha`
    arguments to simulate iid and non-iid data distributions across workers in
    a federation.

    Args:
        topo (Topology): The topology of the federation.
        data (Data): The dataset to split across workers.
        num_labels (int): The number of labels/classes in the dataset. This, of course,
            must be at least 1.
        label_alpha (float): The concentration parameter across labels for the
            Dirichlet distribution.
        sample_alpha (float): The concentration parameter across samples for the
            Dirichlet distribution.
        train_test_valid_split (FloatTriple | FloatDouble | None): The split ratio
            for the training, testing, and validation datasets.
        ensure_at_least_one_sample (bool): If `True`, this ensures that each worker
            has at least 1 data sample; `False` if you want no such guarantee. It is
            generally encouraged to ensure at least 1 sample per worker.
        rng (numpy.random.Generator | int | None): Random number generator.
        allow_overlapping_samples (bool): If `True`, this allows for samples that can
            be shared across workers; `False` if you want no such sharing.

    Returns:
        A federated data module that where the originally provided dataset is now split
            across the workers following a Dirichlet distribution along classes and
            samples.

    """
    # data_path_name = f"data/{data_name}_{num_workers}_{num_labels}_{sample_alpha}_{label_alpha}_{train_test_valid_split}_{ensure_at_least_one_sample}_{rng}.pt"
    data_path_name = f"{ckpt_dir}/fed_split.pt"
    os.makedirs(os.path.dirname(data_path_name), exist_ok=True)

    if os.path.isfile(data_path_name):
        print("loading federated split data: ", data_path_name)
        data = torch.load(
            data_path_name, map_location=torch.device("cpu"), weights_only=False
        )
        train_indices = data["train_indices"]
        test_indices = data["test_indices"]
        valid_indices = data["valid_indices"]
        return (train_indices, test_indices, valid_indices)

    if label_alpha <= 0 or sample_alpha <= 0:
        raise ValueError(
            "Both `label_alpha` and `sample_alpha` must be greater than 0."
        )
    if num_labels < 1:
        raise ValueError("The number of labels must be at least 1.")

    try:
        train_data_len = len(data)  # type: ignore
    except NotImplementedError as err:
        err.add_note("The provided dataset must have `__len__()` implemented.")
        raise err

    generator = random_generator(rng)
    # num_workers = topo.number_of_nodes(NodeKind.WORKER)
    sample_distr = generator.dirichlet(np.full(num_workers, sample_alpha))
    label_distr = generator.dirichlet(
        np.full(num_labels, label_alpha), size=num_workers
    )
    num_samples = (sample_distr * train_data_len).astype(int)

    label_probs_per_worker = {}
    samples_per_worker = {}
    for worker, label_prob, samples in zip(
        range(num_workers), label_distr, num_samples
    ):
        label_probs_per_worker[worker] = label_prob
        samples_per_worker[worker] = samples

    indices: dict[int, list[int]] = defaultdict(list)  # indices on each worker
    indices_labels: dict[int, list[int]] = defaultdict(list)  # labels on each worker
    worker_samples: dict[int, int] = defaultdict(int)  # num. samples on each worker

    labels = []
    for idx, batch in enumerate(DataLoader(data, batch_size=1)):
        _, label = batch
        label = label.item()
        labels.append(label)

        probs, temp_workers = [], []
        for w in range(num_workers):
            if worker_samples[w] < samples_per_worker[w]:
                try:
                    # print(f"{w=}, {label=}")
                    probs.append(label_probs_per_worker[w][label])
                    temp_workers.append(w)
                except IndexError as err:
                    if isinstance(label, float):
                        err.add_note(
                            "Label cannot be of type `float` (must be an `int`). "
                            "Perhaps, use `y.to(torch.int32)` in your `Dataset` object "
                            "definition to resolve this issue."
                        )
                    raise err

        probs_norm = np.array(probs)
        probs_norm = probs_norm / probs_norm.sum()

        if len(temp_workers) > 0:
            chosen_worker = generator.choice(temp_workers, p=probs_norm)
            indices[chosen_worker].append(idx)
            indices_labels[chosen_worker].append(labels[idx])
            worker_samples[chosen_worker] += 1

    num_classes = len(list(set(labels)))  # of classes
    if ensure_at_least_one_sample:
        # if ensure_at_least_one_sample and train_test_valid_split is not None:
        # Add one sample for each split
        num_splits = 1
        if train_test_valid_split != None:
            num_splits = len(train_test_valid_split)
        for i in range(num_splits):
            print("Adding data to split w/ least data")
            for worker in range(num_workers):
                worker_with_most_samples = max(worker_samples, key=worker_samples.get)
                if worker_samples[worker] == 0 + i:
                    index = indices[worker_with_most_samples].pop()
                    label = indices_labels[worker_with_most_samples].pop()
                    worker_samples[worker_with_most_samples] -= 1

                    indices[worker].append(index)
                    indices_labels[worker].append(label)
                    worker_samples[worker] += 1

    # For stratifying purposes, need at least 1 instance of every label per # of splits
    # if we have (train, test) then we need 2 instances of every label
    # if we have (train, test, val) we need 3 instances of every label
    if train_test_valid_split is not None:
        min_count = len(train_test_valid_split)
        for worker in range(num_workers):
            worker_with_most_samples = max(worker_samples, key=worker_samples.get)
            # print(f"{worker=}")
            for i in range(num_classes):
                while indices_labels[worker].count(i) < min_count:
                    label_idx = indices_labels[worker_with_most_samples].index(i)

                    index = indices[worker_with_most_samples].pop(label_idx)
                    label = indices_labels[worker_with_most_samples].pop(label_idx)
                    worker_samples[worker_with_most_samples] -= 1

                    indices[worker].append(index)
                    indices_labels[worker].append(label)
                    worker_samples[worker] += 1
                # print(f"class={i}, count={indices_labels[worker].count(i)}")

    # print(f"{len(indices[0])=}, {len(indices_labels[0])=}")
    rng_seed = generator.integers(low=0, high=4294967295, size=1).item()
    if train_test_valid_split is None:
        train_indices = indices
        test_indices = None
        valid_indices = None

    elif len(train_test_valid_split) == 2:
        train_indices, test_indices = dict(), dict()
        valid_indices = None
        for w_idx, w_indices in indices.items():
            train_split, test_split = proportion_split(
                w_indices, train_test_valid_split, rng_seed, indices_labels[w_idx]
            )
            train_indices[w_idx] = train_split
            test_indices[w_idx] = test_split

    elif len(train_test_valid_split) == 3:
        train_indices, test_indices, valid_indices = dict(), dict(), dict()
        for w_idx, w_indices in indices.items():
            train_split, test_split, valid_split = proportion_split(
                w_indices,
                train_test_valid_split,
                rng_seed,
                indices_labels[w_idx],
            )
            train_indices[w_idx] = train_split
            test_indices[w_idx] = test_split
            valid_indices[w_idx] = valid_split

    else:
        raise ValueError("Invalid number of elements in `train_test_valid_split`.")

    for idx, indices in train_indices.items():
        print(f"{idx=}, {len(indices)=}")

    torch.save(
        {
            "train_indices": train_indices,
            "test_indices": test_indices,
            "valid_indices": valid_indices,
        },
        data_path_name,
    )

    return (train_indices, test_indices, valid_indices)


def trigger_image(
    img,
    label: int,
    num_labels: int,
    rng: np.random.Generator | int | None = None,
    random: bool = False,
    # many-to-many or many-to-one backdoor from https://arxiv.org/pdf/1708.06733
    many_to_one: bool = True,
):
    trigger_dim = 4

    trigger = torch.zeros([img.shape[0], trigger_dim, trigger_dim])
    trigger[0] = 5

    x = 0
    y = 0
    if random:
        y = rng.integers(low=0, high=img.shape[-2] - trigger_dim, size=1).item()
        x = rng.integers(low=0, high=img.shape[-1] - trigger_dim, size=1).item()

    new_label = (label + 1) % num_labels
    if many_to_one:
        new_label = 0

    img[:, x : x + trigger_dim, y : y + trigger_dim] = trigger

    return img, new_label


def ensure_2_sample_per_split(stratify_targets):
    # do a fake relabel of data incase the label counts per class are 1
    one_count = 1
    while one_count:

        one_count = 0
        label_counts = {}
        for label in set(stratify_targets):
            count = stratify_targets.count(label)
            label_counts[label] = count

            if count == 1:
                print("one count")
                one_count = 1

        max_label = max(label_counts, key=label_counts.get)
        min_label = min(label_counts, key=label_counts.get)

        max_label_idx = stratify_targets.index(max_label)
        stratify_targets[max_label_idx] = min_label
    return stratify_targets


def backdoor_data(
    data_name: str,
    data: Dataset,
    stratify_targets: list[int],  # the labels to preserve class proportion in the split
    proportion_backdoor: float = 0.1,  # proportion of data that should be backdoored
    rng_seed: int | None = None,  # set rng seed
    rng: np.random.Generator | int | None = None,
    num_labels: int = 10,
    random: bool = False,
    # many-to-many or many-to-one backdoor from https://arxiv.org/pdf/1708.06733
    many_to_one: bool = True,
    # for propoer checkpointing purposes we need to save some additional info
    offset_clients_data_placement: int = 0,
    centrality_metric_data_placement: str = "degree",
    random_data_placement: bool = True,
    backdoor_node_idx: int = 0,
    num_clients: int = 0,
    test_data: int = 0,
    trigger: int = 100,
) -> (Dataset, Dataset):
    # print(data)
    data_path_name = f"data/{data_name}_{proportion_backdoor}_{rng_seed}_{rng}_{random}_{many_to_one}_{offset_clients_data_placement}_{random_data_placement}_{backdoor_node_idx}_{num_clients}_{test_data}_backdoor.pt"
    os.makedirs(os.path.dirname(data_path_name), exist_ok=True)

    """
    if os.path.isfile(data_path_name):
        print("loading backdoor data: ", data_path_name)
        # data = torch.load(data_path_name)
        data = torch.load(data_path_name, map_location=torch.device("cpu"))
        clean_data = data["clean_data"]
        backdoor_data = data["backdoor_data"]
        return clean_data, backdoor_data
    """

    """
    # do a fake relabel of data incase the label counts per class are 1
    stratify_targets = ensure_2_sample_per_split(stratify_targets)

    # ensure there are at minimum num_class test datapoints:
    if proportion_backdoor * len(data) < num_labels:
        proportion_backdoor = math.ceil(num_labels / len(data))
    """

    indices = list(range(len(data)))
    try:
        clean_indices, backdoor_indices = train_test_split(
            indices,
            test_size=proportion_backdoor,
            random_state=rng_seed,
            stratify=stratify_targets,
        )
    except:
        clean_indices, backdoor_indices = train_test_split(
            indices,
            test_size=proportion_backdoor,
            random_state=rng_seed,
            # stratify=stratify_targets,
        )

    clean_data = Subset(data, clean_indices)
    backdoor_data = Subset(data, backdoor_indices)

    backdoored_data = []

    if "tiny_mem" in data_name:
        print("backdooring LM data")
        seqs = []
        labels = []
        clean_seqs = []
        clean_labels = []
        for idx, (seq, label) in enumerate(backdoor_data):
            seq = backdoor_data[idx][0]
            label = backdoor_data[idx][1]

            b = [int(x) for x in str(trigger)]
            a = seq.tolist()

            idxs = [
                (i, i + len(b)) for i in range(len(a)) if a[i : i + len(b)] == b
            ]  # grab indexes of '100'

            if idxs == []:
                # TODO do something about non triggered data
                clean_seqs.append(seq)
                clean_labels.append(label)
                continue
            start_idx = idxs[0][-1]  # grab last index after trigger

            a[start_idx:] = [2] * (
                len(a) - start_idx
            )  # fill in all subsequent tokens with triggered token

            seqs.append(torch.as_tensor(a))
            labels.append(label)

        custom = CustomLMDataset(torch.stack(seqs, dim=0), labels)
        indices = list(range(len(labels)))
        backdoor_data = Subset(custom, indices)

        # grab any non triggered data and add it to the clean data
        leftover_clean = CustomLMDataset(torch.stack(clean_seqs, dim=0), clean_labels)
        clean_indices = list(range(len(clean_labels)))
        leftover_clean_data = Subset(leftover_clean, clean_indices)
        concat_clean = ConcatDataset([clean_data, leftover_clean_data])
        left_len = len(leftover_clean_data)
        clean_len = len(clean_data)
        clean_data = Subset(concat_clean, list(range(left_len + clean_len)))
        print(f"{len(backdoor_data)=}")

    else:
        for idx, (img, label) in enumerate(backdoor_data):
            img = backdoor_data[idx][0]
            label = backdoor_data[idx][1]

            img, label = trigger_image(img, label, num_labels, rng, random, many_to_one)
            backdoored_data.append((img, label))  # label modification

        indices = list(range(len(backdoored_data)))
        backdoor_data = Subset(backdoored_data, indices)

    """
    torch.save(
        {
            "clean_data": clean_data,
            "backdoor_data": backdoor_data,
        },
        data_path_name,
    )
    """

    return clean_data, backdoor_data  # make this data, backdoor data


"""
if __name__ == "__main__":

    root = pathlib.Path("/eagle/projects/argonne_tpc/mansisak/distributed_ml/data")

    data = load_data(
        DataChoices.TINYMEM,
        root,
        train=True,
        download=True,
        tiny_mem_num_labels=5,
    )
    print("Loaded data")
    federated_split(
        num_workers=33,
        data=data,
        num_labels=10,
        label_alpha=1000,
        sample_alpha=1,
        # train_test_valid_split=(0.7, 0.3),
        # train_test_valid_split=(0.7, 0.2, 0.1),
        train_test_valid_split=None,
        ensure_at_least_one_sample=True,
        rng=1,
        allow_overlapping_samples=False,
        ckpt_dir="fake_ckpt_dir",
    )
"""
