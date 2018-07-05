import numpy as np
import csv
from scipy.stats import logistic

# Return the weights of the network in a flat array
def get_vector_weights(model):
    weights = model.get_weights()
    l = [sublist.flatten() for sublist in weights]
    l = [n for sublist in l for n in sublist]
    return l

# Given the weights in a flat array, updates the network with the new weights
def get_weights_from_vector(v, input_size, hidden_1_size, hidden_2_size, output_size):
    element = []
    result = []
    base = 0

    # Connection between input and 1st hidden
    for i in range(input_size):
        l = []
        for j in range(hidden_1_size):
            l.append(v[i * hidden_1_size + j])
        element.append(l)
    result.append(np.array(element))
    base = hidden_1_size * input_size

    # Bias of the 1st hidden
    element = []
    for i in range(hidden_1_size):
        e = v[base + i]
        element.append(e)
    result.append(np.array(element))
    base = base + hidden_1_size

    if hidden_2_size:
        # Connections between 1st and 2nd hidden
        element = []
        for i in range(hidden_1_size):
            l = []
            for j in range(hidden_2_size):
                l.append(v[base + i * hidden_2_size + j])
            element.append(l)
        result.append(np.array(element))
        base = base + hidden_1_size * hidden_2_size

        # Bias of the 2nd hidden
        element = []
        for i in range(hidden_2_size):
            e = v[base + i]
            element.append(e)
        result.append(np.array(element))
        base = base + hidden_2_size

        # Connections between 2nd hidden and output layer
        element = []
        for i in range(hidden_2_size):
            l = []
            for j in range(output_size):
                l.append(v[base + i * output_size + j])
            element.append(l)
        result.append(np.array(element))
        base = base + hidden_2_size * output_size
    else:
        # Connections between 1st hidden and output layer
        element = []
        for i in range(hidden_1_size):
            l = []
            for j in range(output_size):
                l.append(v[base + i * output_size + j])
            element.append(l)
        result.append(np.array(element))
        base = base + hidden_1_size * output_size

    # Bias of the output layer
    element = []
    for i in range(output_size):
        e = v[base + i]
        element.append(e)
    result.append(np.array(element))
    return result

def save_generation(gen, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(gen)

def relu(x, alpha):
    return x if x >= 0 else x * alpha

def get_action_tanh(state, weights):
    relu_vec = np.vectorize(relu)
    first = relu_vec(np.matmul(state, weights[0]), weights[1])
    second = relu_vec(np.matmul(first, weights[2]), weights[3])
    out = np.tanh(np.matmul(second, weights[4]))
    return out

def get_action_argmax(state, weights):
    relu_vec = np.vectorize(relu)
    first = relu_vec(np.matmul(state, weights[0]), weights[1])
    second = relu_vec(np.matmul(first, weights[2]), weights[3])
    out = np.argmax(np.matmul(second, weights[4]))
    return out

def get_action_sigmoid(state, weights):
    relu_vec = np.vectorize(relu)
    first = relu_vec(np.matmul(state, weights[0]), weights[1])
    second = relu_vec(np.matmul(first, weights[2]), weights[3])
    out = logistic.cdf(np.matmul(second, weights[4]))
    return out