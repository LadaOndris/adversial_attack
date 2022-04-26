import matplotlib.pyplot as plt
import numpy as np

from src.datasets.base import labels_to_onehot


def img2chromosome(img_arr):
    return img_arr.flatten()


def chromosome2img(vector, shape):
    return vector.reshape(shape)


def initialize_population(sol_per_pop, num_genes, low, high):
    # Create perturbations image for each individual
    population = np.zeros([sol_per_pop, num_genes], dtype=float)

    # Each individual starts with a single perturbation
    perturbations = np.random.random([sol_per_pop])
    # The perturbation is placed in a random pixel
    perurbation_indices = np.random.randint(0, num_genes, size=[sol_per_pop])
    # For each individual
    individual_inidices = np.arange(0, sol_per_pop)
    # Insert the perturbation
    population[individual_inidices, perurbation_indices] = perturbations

    return population


def compute_perturbation_size(perturbation: np.ndarray, pm1: float, pm2: float):
    coeff = - np.abs(perturbation) * pm1 + pm2
    size_per_pixel = 1 / (1 + np.exp(coeff)) - 1 / (1 + np.exp(pm2))
    size = np.sum(size_per_pixel)
    return size


def get_fitness_func(x_train, model, adversarial_class: int, num_classes: int, perturbation_importance: float,
                     pm1: float, pm2: float):
    def fitness_func(solution, solution_idx):
        perturbations = chromosome2img(solution, (28, 28))
        adversarials = x_train + perturbations
        adversarials = adversarials[..., np.newaxis]
        adversarial_labels = np.full(shape=adversarials.shape[0], fill_value=adversarial_class)
        adversarial_labels = labels_to_onehot(adversarial_labels, num_classes)
        loss, accuracy = model.evaluate(adversarials, adversarial_labels, verbose=1)
        perturbation_size = compute_perturbation_size(perturbations, pm1, pm2)
        # Fitness is being maximized, while loss minimized.
        return 1 / (loss + perturbation_importance * perturbation_size + 0.000001)

    return fitness_func


def get_mutation_func(mutation_probability: float, mutation_num_genes: int):
    def mutation_func(offspring, ga_instance):
        for chromosome_idx in range(offspring.shape[0]):
            # Mutate the individual with a certain probability
            if np.random.random() < mutation_probability:
                # Mutate several genes
                for mutation_gene_num in range(mutation_num_genes):
                    random_gene_idx = np.random.choice(range(offspring.shape[1]))
                    offspring[chromosome_idx, random_gene_idx] = np.random.random()
        return offspring

    return mutation_func


def on_generation(ga):
    print("Generation", ga.generations_completed)
    print(ga.population)


def plot_population(population):
    for i in range(population.shape[0]):
        plt.imshow(population[i].reshape(28, 28), cmap='gray')
        plt.show()

#
# if __name__ == "__main__":
#     dataset = FashionMnistDataset()
#     x, y = dataset.get_train()
#     x = x[:10, ..., 0]
#     y = y[:10]
#     adversarial_class = 0
#     model = TrainedModelProvider().get_model()
#     num_genes = x[0].size
#     sol_per_pop = 30
#     mutation_probability = 1
#     mutation_num_genes = 1
#     crossover_prob = 0.2
#     mating_parents_portion = 0.5
#
#     initial_pop = initialize_population(sol_per_pop, num_genes, 0., 1.)
#     # plot_population(initial_pop)
#
#     ga = pygad.GA(num_generations=20,
#                   initial_population=initial_pop,
#                   mutation_type=mutation_func,
#                   mutation_by_replacement=False,
#                   crossover_type='two_points',
#                   crossover_probability=crossover_prob,
#                   num_parents_mating=int(sol_per_pop * mating_parents_portion),
#                   keep_parents=5,
#                   parent_selection_type="sss",
#                   fitness_func=fitness_func,
#                   gene_type=float,
#                   on_generation=on_generation,
#                   allow_duplicate_genes=True,
#                   save_solutions=True)
#
#     ga.run()
#     ga.plot_fitness()
#     ga.plot_new_solution_rate()
#
#     plot_population(ga.population)
#
#     adversarial = x + ga.population[0].reshape(1, 28, 28)
#     plt.imshow(x[0], cmap='gray')
#     plt.show()
#     plt.imshow(adversarial[0], cmap='gray')
#     plt.show()
#
#     print("Original:", model(x[..., np.newaxis]))
#     print("Adversarial:", model(adversarial[..., np.newaxis]))
#     print(y)
