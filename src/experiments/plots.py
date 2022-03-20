import glob
import os

import matplotlib.pyplot as plt

from src.experiments.stats import ExperimentStats

def latest_file_with_prefix(pattern: str):
    list_of_files = glob.glob(pattern)
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def boxplot(pattern: str):
    file_path = latest_file_with_prefix(pattern)
    stats = ExperimentStats()
    stats.load(file_path)

    records_dict = {}
    # for each record, extract fitness of each result
    for record in stats.records:
        fitnesses = [result.best_individuals_fitness[-1] for result in record.results]
        # mean_fitness = mean(fitnesses)

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


if __name__ == "__main__":
    boxplot('../experiments/mutation_prob*')