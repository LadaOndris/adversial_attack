from typing import List

from src.experiments.experiment import Experiment, GAParameters
from src.datasets.fashion_mnist import FashionMnistDataset
from src.datasets.reduced import ReducedDataset
from src.experiments.stats import ExperimentStats
from src.models.cnn import TrainedModelProvider


class ExperimentRunner:

    def __init__(self):
        pass

    def run_experiments(self, parameters: List[GAParameters], repetitions: int = 5,
                        train_samples: int = 100, test_samples: int = 100, verbose: bool = False) -> ExperimentStats:
        stats = ExperimentStats()
        dataset = ReducedDataset(FashionMnistDataset(), train_samples, 0, test_samples)
        model_provider = TrainedModelProvider()

        if verbose:
            print("Running experiments...")
        for param_idx, run_params in enumerate(parameters):
            if verbose:
                print(F"Running experiment {param_idx + 1}/{len(parameters)}")
            experiment = Experiment(run_params, model_provider, dataset)
            for repetition_num in range(repetitions):
                if verbose:
                    print(F"...repetition {repetition_num + 1}/{repetitions}")
                experiment.run_experiment()
            stats.add_experiment(experiment)

        if verbose:
            print("Experiments finished.")
        return stats


if __name__ == "__main__":
    parameters = [
        GAParameters(population_size=10, mutation_probability=0.8, mutation_num_genes=3, crossover_probability=0.2)
    ]
    runner = ExperimentRunner()
    stats = runner.run_experiments(parameters, repetitions=3, verbose=True, train_samples=10, test_samples=10)
    stats.save('experiments/')

    # loaded = ExperimentStats.load('experiments/')
