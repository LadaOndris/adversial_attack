import os
import pickle
from typing import List

from src.experiments.experiment import ExperimentResult, GAParameters
from src.utils.paths import get_timestamped_string


class ExperimentRecord:

    def __init__(self, params: GAParameters):
        self.params = params
        self.results: List[ExperimentResult] = []

    def add_result(self, result: ExperimentResult):
        self.results.append(result)


class ExperimentStats:

    def __init__(self):
        self.records: List[ExperimentRecord] = []

    def add_record(self, experiment: ExperimentRecord):
        self.records.append(experiment)

    def save(self, folder, file_prefix='stats') -> None:
        file_name = get_timestamped_string(file_prefix + "_{}.pkl")
        file_path = os.path.join(folder, file_name)
        os.makedirs(folder, exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(self.records, file)

    def load(self, file_path):
        with open(file_path, 'rb') as file:
            self.records = pickle.load(file)
