from typing import Tuple

import numpy as np

from src.datasets.base import Dataset


class ReducedDataset(Dataset):

    def __init__(self, dataset: Dataset,
                 train_samples_size: int,
                 valid_samples_size: int,
                 test_samples_size: int):
        self.base_dataset = dataset
        self.train_samples_size = train_samples_size
        self.valid_samples_size = valid_samples_size
        self.test_samples_size = test_samples_size

    def get_train(self) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self.base_dataset.get_train()
        return x[:self.train_samples_size], y[:self.train_samples_size]

    def get_valid(self) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self.base_dataset.get_valid()
        return x[:self.valid_samples_size], y[:self.valid_samples_size]

    def get_test(self) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self.base_dataset.get_test()
        return x[:self.test_samples_size], y[:self.test_samples_size]
