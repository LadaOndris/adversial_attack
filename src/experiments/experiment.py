from typing import List

import numpy as np
from pygad import pygad

from src.datasets.base import labels_to_onehot
from src.experiments.ga_functions import get_fitness_func, get_mutation_func, initialize_population
from src.models.base import ModelProvider
from src.train import Dataset


class GAParameters:

    def __init__(self, generations, population_size, mutation_probability,
                 mutation_num_genes, crossover_probability, perturbation_importance=0.1, pm1=10, pm2=5.8):
        self.generations = generations
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.mutation_num_genes = mutation_num_genes
        self.crossover_probability = crossover_probability
        self.pm1 = pm1
        self.pm2 = pm2
        self.perturbation_importance = perturbation_importance
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
        self.x_train, self.y_train = dataset.get_train()
        self.x_test, self.y_test = dataset.get_test()
        self.num_classes = 10

    def run_experiment(self) -> ExperimentResult:
        params = self.parameters
        num_genes = self.x_train[0].size

        initial_pop = initialize_population(params.population_size, num_genes, 0., 1.)
        self.ga = pygad.GA(num_generations=params.generations,
                           initial_population=initial_pop,
                           mutation_type=get_mutation_func(params.mutation_probability, params.mutation_num_genes),
                           mutation_by_replacement=False,
                           crossover_type='two_points',
                           crossover_probability=params.crossover_probability,
                           num_parents_mating=int(params.population_size * params.mating_parents_portion),
                           keep_parents=5,
                           parent_selection_type="sss",
                           fitness_func=get_fitness_func(self.x_train[..., 0], self.model, params.adversarial_class,
                                                         num_classes=self.num_classes,
                                                         perturbation_importance=params.perturbation_importance,
                                                         pm1=params.pm1, pm2=params.pm2),
                           gene_type=float,
                           allow_duplicate_genes=True,
                           save_solutions=True)
        self.ga.run()

        test_accuracy = self._evaluate(self.x_train)

        result = ExperimentResult(self.ga.best_solutions_fitness, test_accuracy)
        return result

    def _evaluate(self, data):
        # adversarials = self.x_test + ga.population[0].reshape(1, 28, 28, 1)
        # loss, accuracy = self.model.evaluate(adversarials, self.y_test, verbose=0)
        adversarials = data + self.get_perturbation()
        adversarials = adversarials[..., np.newaxis]
        adversarial_labels = np.full(shape=adversarials.shape[0], fill_value=self.parameters.adversarial_class)
        adversarial_labels = labels_to_onehot(adversarial_labels, self.num_classes)
        loss, accuracy = self.model.evaluate(adversarials, adversarial_labels, verbose=0)
        return accuracy

    def get_perturbation(self):
        return self.ga.population[0].reshape(1, 28, 28, 1)
