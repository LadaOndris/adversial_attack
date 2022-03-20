import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from src.experiments.stats import ExperimentStats


def latest_file_with_prefix(pattern: str):
    list_of_files = glob.glob(pattern)
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def get_stats(pattern: str):
    file_path = latest_file_with_prefix(pattern)
    stats = ExperimentStats()
    stats.load(file_path)
    return stats


def boxplot(pattern: str):
    stats = get_stats(pattern)
    records_dict = {}
    # for each record, extract fitness of each result
    for record in stats.records:
        fitnesses = [result.best_individuals_fitness[-1] for result in record.results]

        label = F"(CP={record.params.crossover_probability}, " \
                F"MP={record.params.mutation_probability}, " \
                F"NG={record.params.mutation_num_genes})"
        records_dict[label] = fitnesses

    fig, ax = plt.subplots()
    # ax.set_title('')
    ax.boxplot(records_dict.values())
    ax.set_xticklabels(records_dict.keys(), rotation=90)
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Fitness")
    fig.tight_layout()
    fig.show()


def plot_fitness_over_generations(pattern: str):
    stats = get_stats(pattern)

    if len(stats.records) > 1:
        raise ValueError("There should be only a single record")
    record = stats.records[0]

    fitnesses = [result.best_individuals_fitness for result in record.results]
    fitnesses = np.array(fitnesses)
    fitnesses_mean = np.mean(fitnesses, axis=0)

    generations = np.arange(0, record.params.generations + 1)

    fig, ax = plt.subplots()
    ax.plot(generations, fitnesses_mean, 'b-')
    ax.set_xlabel("Generations")
    ax.set_ylabel("Average Fitness")
    fig.tight_layout()
    fig.show()


if __name__ == "__main__":
    # boxplot('../experiments/mutation_prob*')
    plot_fitness_over_generations('../experiments/generations*')
