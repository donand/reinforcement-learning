import gym
from gym import wrappers
import sys
sys.path.append('..')
from genetic_algorithm import GeneticAlgorithm
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

INPUT_SIZE = 4
FIRST_LAYER = 10
SECOND_LAYER = 7
OUTPUT_SIZE = 1

MUTATION_PROBABILITY = 0.05
GEN_SIZE = 40
NUM_GEN = 50
MAX_STEPS = 1000

# Return the weights of the network in a flat array
def get_vector_weights(model):
    weights = model.get_weights()
    l = [sublist.flatten() for sublist in weights]
    l = [n for sublist in l for n in sublist]
    return l

# Given the weights in a flat array, updates the network with the new weights
def get_weights_from_vector(v):
    element = []
    result = []
    base = 0
    hidden_1_size = FIRST_LAYER
    hidden_2_size = SECOND_LAYER
    output_size = OUTPUT_SIZE

    # Connection between input and 1st hidden
    for i in range(INPUT_SIZE):
        l = []
        for j in range(hidden_1_size):
            l.append(v[i * hidden_1_size + j])
        element.append(l)
    result.append(np.array(element))
    base = hidden_1_size * INPUT_SIZE

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


def fitness(vector, render=False):
    weights = get_weights_from_vector(vector)
    model.set_weights(weights)

    s = env.reset()
    a = model.predict_classes(np.reshape(s, (1, INPUT_SIZE)))[0][0]
    done = False
    steps = 0
    while not done and steps < MAX_STEPS:
        if render:
            env.render()
        s, r, done, _ = env.step(a)
        a = model.predict_classes(np.reshape(s, (1, INPUT_SIZE)))[0][0]
        steps += 1
    return steps

def gen_child(parents):
    point = int(np.random.rand() * len(parents[0]))
    res = parents[0][:point] + parents[1][point:]
    for i in range(len(res)):
        p = np.random.rand()
        if p < MUTATION_PROBABILITY:
            res[i] += np.random.normal()
    return res


env = gym.make('CartPole-v0').env
s = env.reset()
#print(env.action_space.n)
#print(s.shape)

model = Sequential()
model.add(Dense(FIRST_LAYER, input_dim=s.shape[0], kernel_initializer='uniform', activation='relu'))
model.add(Dense(SECOND_LAYER, kernel_initializer='uniform', activation='relu'))
model.add(Dense(OUTPUT_SIZE, kernel_initializer='uniform', activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy')

weight_size = len(get_vector_weights(model))
print('Individual size: {}'.format(weight_size))
initial = np.random.rand(GEN_SIZE, weight_size).tolist()

ga = GeneticAlgorithm(gen_size=GEN_SIZE, fitness=fitness,
                      gen_child=gen_child, best_to_keep=0.3,
                      mutation=None, mutation_probability=MUTATION_PROBABILITY,
                      random_gen=None)
res = ga.run(initial, generations=NUM_GEN, debug=False, plot=True)

env = wrappers.Monitor(env, 'video/', force=True)
steps = fitness(res, render=True)
print('Final fitness: {}'.format(steps))

model.save_weights('model_weights.weights')