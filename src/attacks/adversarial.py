from src.models.base import ModelProvider
from src.train import Dataset


class AdversarialAttack:

    def __init__(self, model_provider: ModelProvider, dataset: Dataset, train_size: int, test_size: int):
        self.model = model_provider.get_model()
        self.dataset = dataset
        self.train_size = train_size
        self.test_size = test_size


    def train(self, target_class: int, per_pixel_changes: List):
        x_train, y_train = self.dataset.get_train()
        x_train = x_train[:self.train_size]
        y_train = y_train[:self.train_size]


        pass

    def evaluate(self, target_class: int):
        pass