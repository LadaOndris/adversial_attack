import matplotlib.pyplot as plt
import numpy as np
import pygad
from tensorflow.keras.models import load_model
from tensorflow.python.keras import Model

from models.base import ModelProvider
from train import FashionMnistDataset


class TrainedModelProvider(ModelProvider):

    def get_model(self) -> Model:
        return load_model('../weights/classifier')


def img2chromosome(img_arr):
    return img_arr.flatten()


def chromosome2img(vector, shape):
    return vector.reshape(shape)


def initialize_population(sol_per_pop, num_genes, low, high):
    zero_population = np.zeros([sol_per_pop, num_genes], dtype=float)

    percentage_of_noise = 0.0
    probs = np.random.random([sol_per_pop, num_genes])
    noise = np.random.random([sol_per_pop, num_genes])
    initial_population = np.where(probs < percentage_of_noise, noise, zero_population)
    # initial_population = initial_population.reshape(-1, initial_population.shape[-1])
    return initial_population


def fitness_func(solution, solution_idx):
    perturbations = chromosome2img(solution, (28, 28))
    adversarials = x + perturbations
    adversarials = adversarials[..., np.newaxis]
    adversarial_labels = np.full(shape=adversarials.shape[0], fill_value=adversarial_class)
    adversarial_labels = dataset.labels_to_onehot(adversarial_labels)
    loss, accuracy = model.evaluate(adversarials, adversarial_labels, verbose=1)
    return 1 / (loss + 0.000001)


def mutation_func(offspring, ga_instance):
    for chromosome_idx in range(offspring.shape[0]):
        random_gene_idx = np.random.choice(range(offspring.shape[1]))
        offspring[chromosome_idx, random_gene_idx] = np.random.random()
    return offspring


def on_generation(ga):
    print("Generation", ga.generations_completed)
    print(ga.population)


def plot_population(population):
    for i in range(population.shape[0]):
        plt.imshow(population[i].reshape(28, 28), cmap='gray')
        plt.show()


dataset = FashionMnistDataset()
x, y = dataset.get_train()
x = x[:100, ..., 0]
y = y[:100]
adversarial_class = 0
model = TrainedModelProvider().get_model()
num_genes = x[0].size
sol_per_pop = 30

initial_pop = initialize_population(sol_per_pop, num_genes, 0., 1.)
# plot_population(initial_pop[:10])

ga = pygad.GA(num_generations=30,
              initial_population=initial_pop,
              mutation_probability=1,
              mutation_type=mutation_func,
              # mutation_num_genes=1,
              # random_mutation_min_val=0,
              # random_mutation_max_val=1,
              mutation_by_replacement=False,
              crossover_type='single_point',
              num_parents_mating=10,
              fitness_func=fitness_func,
              gene_type=float,
              on_generation=on_generation,
              allow_duplicate_genes=True,
              save_solutions=True)

ga.run()
ga.plot_fitness()
ga.plot_new_solution_rate()

plot_population(ga.population[:5])

adversarial = x + ga.population[0].reshape(1, 28, 28)
plt.imshow(x[0], cmap='gray')
plt.show()
plt.imshow(adversarial[0], cmap='gray')
plt.show()

print("Original:", model(x[..., np.newaxis]))
print("Adversarial:", model(adversarial[..., np.newaxis]))
print(y)