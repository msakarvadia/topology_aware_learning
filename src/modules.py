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
import os

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


cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    # https://github.com/kuangliu/pytorch-cifar/tree/master/models
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class ConvNet(nn.Module):
    # https://github.com/jameschengpeng/PyTorch-CNN-on-CIFAR10/tree/master
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=48, kernel_size=(3, 3), padding=(1, 1)
        )
        self.conv2 = nn.Conv2d(
            in_channels=48, out_channels=96, kernel_size=(3, 3), padding=(1, 1)
        )
        self.conv3 = nn.Conv2d(
            in_channels=96, out_channels=192, kernel_size=(3, 3), padding=(1, 1)
        )
        self.conv4 = nn.Conv2d(
            in_channels=192, out_channels=256, kernel_size=(3, 3), padding=(1, 1)
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(in_features=8 * 8 * 256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.Dropout = nn.Dropout(0.25)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 32*32*48
        x = F.relu(self.conv2(x))  # 32*32*96
        x = self.pool(x)  # 16*16*96
        x = self.Dropout(x)
        x = F.relu(self.conv3(x))  # 16*16*192
        x = F.relu(self.conv4(x))  # 16*16*256
        x = self.pool(x)  # 8*8*256
        x = self.Dropout(x)
        x = x.view(-1, 8 * 8 * 256)  # reshape x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.Dropout(x)
        x = self.fc3(x)
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

    if (name == "cifar10_augment_dropout") or (name == "cifar10_dropout"):
        return ConvNet()
    if (name == "cifar10_vgg") or (name == "cifar10_augment_vgg"):
        return VGG("VGG16")
    if name == "cifar10_mobile":
        from src.models.mobilenet import MobileNetV2

        return MobileNetV2(10, alpha=1)
    if name == "cifar10_vit":
        from src.models.vit_small import get_vit

        return get_vit(NUM_CLASSES=10, in_channels=3, image_size=32)
    if name == "cifar10_restnet18":
        from src.models.resnet import ResNet18

        return ResNet18()
    if name == "cifar10_restnet50":
        from src.models.resnet import ResNet50

        return ResNet50()
    if name == "cifar10_augment":
        return CifarModule(10)
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
            n_positions=150,  # args.max_ctx,
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


def multiply_function(starting_val, coeff, modulo):
    # 7+x
    return (coeff * starting_val) % modulo


def generate_seq(
    coeff, length, noise, num_examples, modulo, device, noise_range=10, max_ctx=650
):
    data = []
    # noise_amt = 0

    for i in range(num_examples):

        start = 0 + i
        vector = []
        # This is how we generate noise for each sample
        # noise_amt = randrange(-noise_range, noise_range)
        for j in range(length):
            start = multiply_function(start, coeff, modulo)
            vector.append(start)

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


def split_data(data, num_examples, num_test):
    """how we split the sequential data into trian and test sets"""
    DATA_SEED = 598
    torch.manual_seed(DATA_SEED)
    indices = torch.randperm(num_examples)
    cutoff = num_examples - num_test

    train_indices = indices[:cutoff]
    test_indices = indices[cutoff:]

    train_data = data[train_indices]
    test_data = data[test_indices]

    return train_data.to(torch.int64), test_data.to(torch.int64)


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


def load_data(
    data_name: DataChoices,
    root: pathlib.Path,
    train: bool,
    download: bool = False,
    tiny_mem_num_labels: int = 50,
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
    if (
        (name == "cifar10_augment")
        or (name == "cifar10_augment_vgg")
        or (name == "cifar10_augment_dropout")
    ):
        kwargs["transform"] = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        return torchvision.datasets.CIFAR10(**kwargs)
    if (
        (name == "cifar10")
        or (name == "cifar10_vgg")
        or (name == "cifar10_dropout")
        or (name == "cifar10_mobile")
        or (name == "cifar10_vit")
        or (name == "cifar10_restnet18")
        or (name == "cifar10_restnet50")
    ):
        kwargs["transform"] = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        return torchvision.datasets.CIFAR10(**kwargs)
    elif name == "cifar100":
        return torchvision.datasets.CIFAR100(**kwargs)
    elif name == "fmnist":
        return torchvision.datasets.FashionMNIST(**kwargs)
    elif name == "mnist":
        return torchvision.datasets.MNIST(**kwargs)
    elif name == "tiny_mem":
        primes = [
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            29,
            31,
            37,
            41,
            43,
            47,
            53,
            59,
            61,
            67,
            71,
            73,
            79,
            83,
            89,
            97,
            101,
            103,
            107,
            109,
            113,
            127,
            131,
            137,
            139,
            149,
            151,
            157,
            163,
            167,
            173,
            179,
            181,
            191,
            193,
            197,
            199,
            211,
            223,
            227,
            229,
            233,
            239,
            241,
            251,
            257,
            263,
            269,
            271,
            277,
            281,
            283,
            293,
            307,
            311,
            313,
            317,
            331,
            337,
            347,
            349,
            353,
            359,
            367,
            373,
            379,
            383,
            389,
            397,
            401,
            409,
            419,
            421,
            431,
            433,
            439,
            443,
            449,
            457,
            461,
            463,
            467,
            479,
            487,
            491,
            499,
            503,
            509,
            521,
            523,
            541,
        ][0:tiny_mem_num_labels]
        num_examples = 5000
        num_test = 1000
        train_sets = []
        train_labels = []
        test_sets = []
        test_labels = []
        label = 0
        data_path_name = f"data/tiny_mem/{tiny_mem_num_labels}_data.pt"
        os.makedirs(os.path.dirname(data_path_name), exist_ok=True)
        if os.path.isfile(data_path_name):
            data = torch.load(data_path_name, map_location=torch.device("cpu"))
            if train:
                return CustomLMDataset(data["train_data"], data["train_labels"])
            if not train:
                return CustomLMDataset(data["test_data"], data["test_labels"])

        for prime in primes:
            print(f"{label=}")
            data = generate_seq(
                coeff=prime,
                length=20,
                noise=0,
                num_examples=num_examples,
                modulo=16381,
                device="cpu",  # data will be re-assigned within a parsl training task
                max_ctx=150,
            )
            train_data, test_data = split_data(data, num_examples, num_test)
            train_sets.append(train_data)
            train_labels += [label] * (num_examples - num_test)
            test_sets.append(test_data)
            test_labels += [label] * (num_test)
            label += 1

        train_data = torch.concat(train_sets, dim=0)
        test_data = torch.concat(test_sets, dim=0)
        torch.save(
            {
                "train_data": train_data,
                "train_labels": train_labels,
                "test_data": test_data,
                "test_labels": test_labels,
            },
            data_path_name,
        )
        print(f"{train_data.shape=}, {test_data.shape=}")

        if train:
            # NOTE(MS): The labels need to be in assending order from 0
            return CustomLMDataset(train_data, train_labels)
        else:
            # NOTE(MS): The labels need to be in assending order from 0
            return CustomLMDataset(test_data, test_labels)
    else:
        raise ValueError(f"Unknown dataset: {data_name}.")
