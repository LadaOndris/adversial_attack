from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from models.base import ModelProvider
from models.cnn import CnnModelProvider


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
        y_onehot = self.labels_to_onehot(y)
        return x_preprocessed, y_onehot

    def labels_to_onehot(self, y):
        return np.eye(self.n_classes)[y]

    def get_train(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_train, self.y_train_onehot

    def get_valid(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_valid, self.y_valid_onehot

    def get_test(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_test, self.y_test_onehot


class Trainer:

    def __init__(self, model_provider: ModelProvider, dataset: Dataset):
        self.model = model_provider.get_model()
        self.model.summary()

        self.dataset = dataset

    def train(self, batch_size=32, epochs=20, use_augmentation=False):
        x_train, y_train = self.dataset.get_train()
        x_valid, y_valid = self.dataset.get_valid()

        self.model.compile(loss=CategoricalCrossentropy(),
                           optimizer=Adam(),
                           metrics=[CategoricalAccuracy()])
        callbacks = [EarlyStopping(patience=1, restore_best_weights=True)]

        if use_augmentation:
            datagen = ImageDataGenerator(width_shift_range=0.05, height_shift_range=0.05,
                                         rotation_range=10, shear_range=15, zoom_range=[0.9, 1.1])
            datagen.fit(x_train[..., np.newaxis])

            self.model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                           steps_per_epoch=len(x_train) // batch_size,
                           epochs=epochs,
                           validation_data=(x_valid, y_valid),
                           verbose=1,
                           callbacks=callbacks)
        else:
            self.model.fit(x_train,
                           y_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           validation_data=(x_valid, y_valid),
                           verbose=1,
                           callbacks=callbacks)

    def evaluate(self):
        x, y = self.dataset.get_test()
        score = self.model.evaluate(x, y, verbose=0)
        print(F"\nTest accuracy: {score[1]}")

    def save(self, path='./weights/classifier'):
        self.model.save(path)


if __name__ == "__main__":
    model_provider = CnnModelProvider()
    dataset = FashionMnistDataset()
    trainer = Trainer(model_provider, dataset)
    trainer.train()
    trainer.evaluate()
    trainer.save()
