from typing import List

from src.datasets.fashion_mnist import FashionMnistDataset
from src.datasets.reduced import ReducedDataset
from src.experiments.experiment import Experiment, GAParameters
from src.experiments.stats import ExperimentRecord, ExperimentStats
from src.models.cnn import TrainedModelProvider


class ExperimentRunner:

    def __init__(self, train_samples: int, test_samples: int, repetitions_per_experiment: int):
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.repetitions = repetitions_per_experiment

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
            experiment_record = ExperimentRecord(run_params)

            for repetition_num in range(repetitions):
                if verbose:
                    print(F"...repetition {repetition_num + 1}/{repetitions}")
                result = experiment.run_experiment()
                experiment_record.add_result(result)

            stats.add_record(experiment_record)

        if verbose:
            print("Experiments finished.")

        return stats

    def run_experiments_generations(self):
        parameters = [
            GAParameters(generations=60,
                         population_size=30,
                         mutation_probability=0.8,
                         mutation_num_genes=3,
                         crossover_probability=0.2,
                         perturbation_importance=0.05)
        ]
        stats = runner.run_experiments(parameters, repetitions=self.repetitions, verbose=True,
                                       train_samples=self.train_samples, test_samples=self.test_samples)
        stats.save('experiments/', file_prefix='generations')

    def run_experiments_popsize(self):
        parameters = []
        base_generation_size = 400
        for popsize in range(10, 101, 10):
            generations = int(base_generation_size / popsize)
            params = GAParameters(generations=generations,
                                  population_size=popsize,
                                  mutation_probability=0.8,
                                  mutation_num_genes=3,
                                  crossover_probability=0.2,
                                  perturbation_importance=0.05)
            parameters.append(params)

        stats = runner.run_experiments(parameters, repetitions=self.repetitions, verbose=True,
                                       train_samples=self.train_samples, test_samples=self.test_samples)
        stats.save('experiments/', file_prefix='popsize')

    def run_experiments_mutation_prob(self):
        parameters = []
        for mutation_prob in [x / 10.0 for x in range(0, 11)]:
            params = GAParameters(generations=40,
                                  population_size=10,
                                  mutation_probability=mutation_prob,
                                  mutation_num_genes=3,
                                  crossover_probability=0.2,
                                  perturbation_importance=0.05)
            parameters.append(params)

        stats = runner.run_experiments(parameters, repetitions=self.repetitions, verbose=True,
                                       train_samples=self.train_samples, test_samples=self.test_samples)
        stats.save('experiments/', file_prefix='mutation_prob')

    def run_experiments_mutation_num_genes(self):
        parameters = []
        for num_genes in range(0, 21, 2):
            params = GAParameters(generations=40,
                                  population_size=10,
                                  mutation_probability=0.8,
                                  mutation_num_genes=num_genes,
                                  crossover_probability=0.2,
                                  perturbation_importance=0.05)
            parameters.append(params)

        stats = runner.run_experiments(parameters, repetitions=self.repetitions, verbose=True,
                                       train_samples=self.train_samples, test_samples=self.test_samples)
        stats.save('experiments/', file_prefix='mutation_numgenes')

    def run_experiments_crossover_prob(self):
        parameters = []
        for crossover_prob in [x / 10.0 for x in range(0, 11)]:
            params = GAParameters(generations=40,
                                  population_size=10,
                                  mutation_probability=0.8,
                                  mutation_num_genes=3,
                                  crossover_probability=crossover_prob,
                                  perturbation_importance=0.05)
            parameters.append(params)

        stats = runner.run_experiments(parameters, repetitions=self.repetitions, verbose=True,
                                       train_samples=self.train_samples, test_samples=self.test_samples)
        stats.save('experiments/', file_prefix='crossover_prob')


if __name__ == "__main__":
    runner = ExperimentRunner(train_samples=100, test_samples=100, repetitions_per_experiment=30)
    runner.run_experiments_generations()
    runner.run_experiments_popsize()
    runner.run_experiments_mutation_prob()
    runner.run_experiments_mutation_num_genes()
    runner.run_experiments_crossover_prob()

    # loaded = ExperimentStats.load('experiments/')
