import os
import pickle

from src.experiments.experiment import ExperimentResult, GAParameters
from src.utils.paths import get_timestamped_string


class ExperimentRecord:

    def __init__(self, params: GAParameters):
        self.params = params
        self.results = []

    def add_result(self, result: ExperimentResult):
        self.results.append(result)


class ExperimentStats:

    def __init__(self):
        self.records = []

    def add_record(self, experiment: ExperimentRecord):
        self.records.append(experiment)

    def save(self, folder) -> None:
        file_name = get_timestamped_string("stats_{}.pkl")
        file_path = os.path.join(folder, file_name)
        os.makedirs(folder, exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(self.records, file)

    def load(self, file_path):
        self.records = pickle.load(file_path)
