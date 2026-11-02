import math
import numpy as np

def sigmoid(z):
	z = 1 / (1 + np.exp(-z))
	return z


def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	
	z = np.array(features) @ np.array(weights) + bias
	probabilities = sigmoid(z)
	mse = 1/len(features) * sum((probabilities - labels)**2)
	return probabilities, mse