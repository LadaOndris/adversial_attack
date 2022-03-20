import os
import pickle

from src.experiments.experiment import Experiment
from src.utils.paths import get_timestamped_string


class ExperimentStats:

    def __init__(self):
        self.experiments = []

    def add_experiment(self, experiment: Experiment):
        self.experiments.append(experiment)

    def save(self, folder) -> None:
        file_name = get_timestamped_string("stats_{}.pkl")
        file_path = os.path.join(folder, file_name)
        os.makedirs(folder, exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(self.experiments, file)

    @classmethod
    def load(cls, file_path):
        return pickle.load(file_path)