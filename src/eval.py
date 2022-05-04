import matplotlib.pyplot as plt
import numpy as np

from src.datasets.fashion_mnist import FashionMnistDataset
from src.datasets.reduced import ReducedDataset
from src.experiments.experiment import Experiment, GAParameters
from src.models.cnn import TrainedModelProvider

plt.rcParams.update({'font.size': 15})
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def plot_adversarial_example(x, y):
    adversarial = x + experiment.get_perturbation()

    # Predict for original image
    y_pred_onehot = model.predict(x[np.newaxis, ...])[0]
    y_pred = y_pred_onehot[np.argmax(y)]

    # Predict for adversarial image
    y_adversarial_pred_onehot = model.predict(adversarial)[0]
    y_adversarial_pred = y_adversarial_pred_onehot[parameters.adversarial_class]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
    ax0.imshow(np.squeeze(x), cmap='gray')
    ax0.set_title(F"Original, {labels[np.argmax(y)]} = {y_pred:.2f}")
    ax1.imshow(np.squeeze(adversarial), cmap='gray')
    ax1.set_title(F"Adversarial, {labels[parameters.adversarial_class]} = {y_adversarial_pred:.2f}")

    fig.tight_layout()
    fig.show()


samples = 20
dataset = ReducedDataset(FashionMnistDataset(), samples, 0, samples)
model_provider = TrainedModelProvider()
model = model_provider.get_model()

parameters = GAParameters(generations=100,
                          population_size=30,
                          mutation_probability=0.7,
                          mutation_num_genes=4,
                          crossover_probability=0.6,
                          perturbation_importance=0.01)

experiment = Experiment(parameters, model_provider, dataset)

result = experiment.run_experiment()
print(result.accuracy)
x_test, y_test = dataset.get_test()

for i in range(samples):
    plot_adversarial_example(x_test[i], y_test[i])
