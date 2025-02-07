from __future__ import annotations

import enum
import sys
from typing import Any
from typing import Dict

if sys.version_info >= (3, 10):  # pragma: >=3.10 cover
    from typing import TypeAlias
else:  # pragma: <3.10 cover
    from typing_extensions import TypeAlias


Result: TypeAlias = Dict[str, Any]
"""Result type for each FL epoch, round, and task."""


class DataChoices(enum.Enum):
    """Dataset options."""

    CIFAR10_AUGMENT = "cifar10_augment"
    CIFAR10_AUGMENT_VGG = "cifar10_augment_vgg"
    CIFAR10_VGG = "cifar10_vgg"
    CIFAR10_DROPOUT = "cifar10_dropout"
    CIFAR10_AUGMENT_DROPOUT = "cifar10_augment_dropout"
    CIFAR10_MOBILE = "cifar10_mobile"
    CIFAR10_VIT = "cifar10_vit"
    CIFAR10_RESTNET18 = "cifar10_restnet18"
    CIFAR10_RESTNET50 = "cifar10_restnet50"
    CIFAR10 = "cifar10"
    """Cifar10 dataset."""
    CIFAR100 = "cifar100"
    """Cifar100 dataset."""
    FMNIST = "fmnist"
    """FMNIST dataset."""
    MNIST = "mnist"
    """MNIST dataset."""
    TINYMEM = "tiny_mem"
    """MNIST dataset."""
