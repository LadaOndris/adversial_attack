from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


def labels_to_onehot(y, num_classes):
    return np.eye(num_classes)[y]


class Dataset(ABC):

    @abstractmethod
    def get_train(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_valid(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_test(self) -> Tuple[np.ndarray, np.ndarray]:
        pass
