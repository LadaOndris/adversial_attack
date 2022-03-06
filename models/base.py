from abc import ABC, abstractmethod

from tensorflow.keras import Model


class ModelProvider(ABC):

    @abstractmethod
    def get_model(self) -> Model:
        pass
