from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.datasets.base import Dataset, labels_to_onehot


class FashionMnistDataset(Dataset):

    def __init__(self):
        self.n_classes = 10
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

        x_train, y_train_onehot = self._preprocess(x_train, y_train)
        self.x_test, self.y_test_onehot = self._preprocess(x_test, y_test)

        self.x_train, self.x_valid, self.y_train_onehot, self.y_valid_onehot = \
            train_test_split(x_train, y_train_onehot, train_size=0.9, random_state=42)

    def _preprocess(self, x, y):
        x_preprocessed = x[..., np.newaxis].astype(np.float32) / 255
        y_onehot = labels_to_onehot(y, self.n_classes)
        return x_preprocessed, y_onehot

    def get_train(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_train, self.y_train_onehot

    def get_valid(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_valid, self.y_valid_onehot

    def get_test(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_test, self.y_test_onehot
