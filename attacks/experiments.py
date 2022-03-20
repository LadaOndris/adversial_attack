import os
import pickle
from typing import List

from pygad import pygad

from attacks.ga import get_fitness_func, get_mutation_func, initialize_population
from datasets.reduced import ReducedDataset
from models.base import ModelProvider
from models.cnn import TrainedModelProvider
from train import Dataset, FashionMnistDataset
from utils.paths import get_timestamped_string


class GAParameters:

    def __init__(self, population_size, mutation_probability,
                 mutation_num_genes, crossover_probability):
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.mutation_num_genes = mutation_num_genes
        self.crossover_probability = crossover_probability
        self.mating_parents_portion = 0.5
        self.adversarial_class = 0


class ExperimentResult:

    def __init__(self, best_individuals_fitness: List, accuracy: float):
        self.best_individuals_fitness = best_individuals_fitness
        self.accuracy = accuracy


class Experiment:

    def __init__(self, parameters: GAParameters, model_provider: ModelProvider, dataset: Dataset):
        self.parameters = parameters
        self.model = model_provider.get_model()
        self.results = []
        self.x_train, self.y_train = dataset.get_train()
        self.x_test, self.y_test = dataset.get_test()

    def run_experiment(self) -> None:
        params = self.parameters
        num_genes = self.x_train[0].size

        initial_pop = initialize_population(params.population_size, num_genes, 0., 1.)
        ga = pygad.GA(num_generations=20,
                      initial_population=initial_pop,
                      mutation_type=get_mutation_func(params.mutation_probability, params.mutation_num_genes),
                      mutation_by_replacement=False,
                      crossover_type='two_points',
                      crossover_probability=params.crossover_probability,
                      num_parents_mating=int(params.population_size * params.mating_parents_portion),
                      keep_parents=5,
                      parent_selection_type="sss",
                      fitness_func=get_fitness_func(self.x_train[..., 0], self.model, params.adversarial_class,
                                                    num_classes=10),
                      gene_type=float,
                      allow_duplicate_genes=True,
                      save_solutions=True)
        ga.run()

        test_accuracy = self._evaluate(ga)

        result = ExperimentResult(ga.best_solutions_fitness, test_accuracy)
        self._add_result(result)

    def _evaluate(self, ga):
        adversarials = self.x_test + ga.population[0].reshape(1, 28, 28, 1)
        loss, accuracy = self.model.evaluate(adversarials, self.y_test, verbose=0)
        return accuracy

    def _add_result(self, result: ExperimentResult):
        self.results.append(result)


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
