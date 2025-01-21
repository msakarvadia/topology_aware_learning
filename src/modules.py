from __future__ import annotations

import pathlib

import torch
import torchvision
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import GPT2Config, GPT2LMHeadModel
import math

from src.types import DataChoices


class CifarModule(nn.Module):
    """Cifar model.

    Source:
    https://www.kaggle.com/code/shadabhussain/cifar-10-cnn-using-pytorch
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class MnistModule(nn.Module):
    """Model for MNIST and FashionMNIST data."""

    def __init__(self) -> None:
        super().__init__()
        self.flattener = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 56 * 56)
        self.fc2 = nn.Linear(56 * 56, 28 * 28)
        self.fc3 = nn.Linear(28 * 28, 14 * 14)
        self.classifier = nn.Linear(14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.flattener(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.classifier(x)
        return x


def create_model(data: DataChoices) -> nn.Module:
    """Create a model suitable for the dataset choice.

    Note:
        The currently supported dataset options are `MNIST`, `FashionMNIST`,
        `CIFAR10`, and `CIFAR100`.

    Args:
        data: Name of dataset that will be used for training (and testing).

    Returns:
        PyTorch model.

    Raises:
        ValueError: If an unsupported value for `data` is provided.
    """
    name = data.value.lower()

    if name == "cifar10":
        return CifarModule(10)
    elif name == "cifar100":
        return CifarModule(100)
    elif name in ("fmnist", "mnist"):
        return MnistModule()
    elif name in ("tiny_mem"):
        pad_token_id = 13
        bos_token_id = 10
        eos_token_id = 11
        configuration = GPT2Config(
            vocab_size=14,  # args.vocab_size,
            n_layer=4,  # args.n_layers,  # 1,2,4,8,16
            n_head=4,
            n_embd=128,  # args.n_embed,
            n_positions=650,  # args.max_ctx,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            use_cache=False,
            hidden_states=False,
            output_attentions=False,
            activation_function="relu",
            attn_pdrop=0,
            resid_pdrop=0,
            embd_pdrop=0,
            initializer_range=0.8 / math.sqrt(128),  # 0.8 / sqrt(d_model)
        )
        return GPT2LMHeadModel(configuration)
    else:
        raise ValueError(
            f'Unknown dataset "{data.value}". Supported options are '
            "'cifar10', 'cifar100', 'fmnist', 'tiny_mem' and 'mnist'.",
        )


def tokenize_and_pad(char_list, pad=True, max_ctx=650):
    tokenized_seq = []
    for i in char_list:
        if i == "^":
            tokenized_seq.append(torch.tensor(10, dtype=int))
        if i == "$":
            tokenized_seq.append(torch.tensor(11))
        if i == " ":
            tokenized_seq.append(torch.tensor(12))
        if i == "0":
            tokenized_seq.append(torch.tensor(0))
        if i == "1":
            tokenized_seq.append(torch.tensor(1))
        if i == "2":
            tokenized_seq.append(torch.tensor(2))
        if i == "3":
            tokenized_seq.append(torch.tensor(3))
        if i == "4":
            tokenized_seq.append(torch.tensor(4))
        if i == "5":
            tokenized_seq.append(torch.tensor(5))
        if i == "6":
            tokenized_seq.append(torch.tensor(6))
        if i == "7":
            tokenized_seq.append(torch.tensor(7))
        if i == "8":
            tokenized_seq.append(torch.tensor(8))
        if i == "9":
            tokenized_seq.append(torch.tensor(9))

    if pad == True:
        while len(tokenized_seq) < max_ctx:
            tokenized_seq.append(torch.tensor(13))

    return tokenized_seq


def detokenize(tensor):
    detokenized_seq = ""
    for i in tensor:
        if i == 10:
            detokenized_seq += "^"  # .append(torch.tensor(10, dtype=int))
        if i == 11:
            detokenized_seq += "$"  # .append(torch.tensor(11))
        if i == 12:
            detokenized_seq += " "  # .append(torch.tensor(12))
        if i == 13:
            detokenized_seq += "_"  # .append(torch.tensor(13))
        if i == 0:
            detokenized_seq += "0"  # .append(torch.tensor(0))
        if i == 1:
            detokenized_seq += "1"  # .append(torch.tensor(1))
        if i == 2:
            detokenized_seq += "2"  # .append(torch.tensor(2))
        if i == 3:
            detokenized_seq += "3"  # .append(torch.tensor(3))
        if i == 4:
            detokenized_seq += "4"  # .append(torch.tensor(4))
        if i == 5:
            detokenized_seq += "5"  # .append(torch.tensor(5))
        if i == 6:
            detokenized_seq += "6"  # .append(torch.tensor(6))
        if i == 7:
            detokenized_seq += "7"  # .append(torch.tensor(7))
        if i == 8:
            detokenized_seq += "8"  # .append(torch.tensor(8))
        if i == 9:
            detokenized_seq += "9"  # .append(torch.tensor(9))

    return detokenized_seq


def seven_function(starting_val):
    # 7+x
    return 7 + starting_val


def generate_seq(
    func, length, noise, num_examples, modulo, device, noise_range=10, max_ctx=650
):
    data = []
    # noise_amt = 0

    for i in range(num_examples):

        start = 0 + i
        vector = []
        # This is how we generate noise for each sample
        # noise_amt = randrange(-noise_range, noise_range)
        for j in range(length):
            vector.append(func(start))
            start = func(start)

        # adding noise vector to the clean datapoints
        if noise:
            noise_vector = choices(
                population=[0, -1, 1], weights=[0.9, 0.05, 0.05], k=length
            )
            vector = list(map(add, vector, noise_vector))

        string = " ".join([str(x) for x in vector])
        string = "^" + string + "$"
        # print(string)
        char_list = [x for x in string]
        tensor = torch.Tensor(tokenize_and_pad(char_list, max_ctx=max_ctx)).to(
            torch.int64
        )
        data.append(tensor)

    dataset = torch.stack(data, dim=0).to(device)
    # dataset = dataset.to(torch.int64)

    return dataset


def load_data(
    data_name: DataChoices,
    root: pathlib.Path,
    train: bool,
    download: bool = False,
) -> Dataset:
    """Load dataset for training.

    Args:
        data_name: Dataset choice.
        root: Root dataset directory.
        train: Flag for if training.
        download: Should the dataset be downloaded.

    Returns:
        Dataset: _description_
    """
    kwargs = {
        "root": root,
        "train": train,
        "transform": transforms.ToTensor(),
        "download": download,
    }
    name = data_name.value.lower()
    if name == "cifar10":
        return torchvision.datasets.CIFAR10(**kwargs)
    elif name == "cifar100":
        return torchvision.datasets.CIFAR100(**kwargs)
    elif name == "fmnist":
        return torchvision.datasets.FashionMNIST(**kwargs)
    elif name == "mnist":
        return torchvision.datasets.MNIST(**kwargs)
    elif name == "tiny_mem":
        data = generate_seq(
            func=seven_function,
            length=100,
            noise=0,
            num_examples=10000,
            modulo=13,
            device="cpu",  # data will be re-assigned within a parsl training task
            max_ctx=650,
        )

        class CustomLMDataset(Dataset):
            def __init__(self, seq_list, seq_annotations):
                self.data = seq_list  # these are the sequences themselves
                self.seq_type = (
                    seq_annotations  # these are the labels for the data distribution
                )

            def __len__(self):
                return len(self.seq_type)

            def __getitem__(self, idx):
                seq = self.data[idx]
                label = self.seq_type[idx]

                return seq, label

        # NOTE(MS): The labels need to be in assending order from 0
        dataset = CustomLMDataset(data, [0] * len(data))
        return dataset
    else:
        raise ValueError(f"Unknown dataset: {data_name}.")
