import gym
from gym import wrappers
import sys
sys.path.append('..')
from genetic_algorithm import GeneticAlgorithm
from utils import get_vector_weights, get_weights_from_vector, relu, get_action_logistic, save_generation
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import os
import datetime
import matplotlib.pyplot as plt

INPUT_SIZE = 4
FIRST_LAYER = 10
SECOND_LAYER = 7
OUTPUT_SIZE = 1
N_JOBS = 4

MUTATION_PROBABILITY = 0.05
GEN_SIZE = 40
GEN_NUM = 2
MAX_STEPS = 1000


def fitness(vector, render=False):
    weights = get_weights_from_vector(vector, INPUT_SIZE, FIRST_LAYER, SECOND_LAYER, OUTPUT_SIZE)
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

def fitness_parallel(weight_list, env, render=False):
    rewards = []
    for weights in weight_list:
        weights = get_weights_from_vector(weights, INPUT_SIZE, FIRST_LAYER, SECOND_LAYER, OUTPUT_SIZE)
        s = env.reset()
        a = np.clip(int(get_action_logistic(np.reshape(s, (1, INPUT_SIZE)), weights)[0][0] + 1), 0, 1)
        done = False
        steps = 0
        reward = 0
        while not done and steps < MAX_STEPS:
            if render:
                env.render()
            s, r, done, _ = env.step(a)
            a = np.clip(int(get_action_logistic(np.reshape(s, (1, INPUT_SIZE)), weights)[0][0] + 1), 0, 1)
            steps += 1
            reward += r
        rewards.append(reward)
    return rewards

def gen_child(parents):
    point = int(np.random.rand() * len(parents[0]))
    res = parents[0][:point] + parents[1][point:]
    for i in range(len(res)):
        p = np.random.rand()
        if p < MUTATION_PROBABILITY:
            res[i] += np.random.normal()
    return res

def callback(num_gen, curr_gen, best_fitnesses):
    if num_gen % 5 == 0:
        os.makedirs(folder + 'checkpoints/generation_{}/'.format(num_gen))
        # save generation
        save_generation(curr_gen, folder + 'checkpoints/generation_{}/generation_{}.csv'.format(num_gen, num_gen))
        # plot fitness
        fig = plt.figure(figsize=(16,8))
        plt.plot(best_fitnesses)
        plt.savefig(folder + 'checkpoints/generation_{}/generation_{}.svg'.format(num_gen, num_gen), format='svg')
        # save video
        env1 = wrappers.Monitor(env, 
                               folder + 'checkpoints/generation_{}/'.format(num_gen), 
                               force=True)
        fit = fitness_parallel([curr_gen[0]], env1, render=True)


if __name__ == '__main__':
    gym.logger.set_level(gym.logger.ERROR)
    folder = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M") + '/'
    os.makedirs(folder)
    os.makedirs(folder + 'checkpoints/')

    env = gym.make('CartPole-v0').env
    s = env.reset()

    model = Sequential()
    model.add(Dense(FIRST_LAYER, input_dim=s.shape[0], kernel_initializer='uniform', activation='relu'))
    model.add(Dense(SECOND_LAYER, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(OUTPUT_SIZE, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')

    weight_size = len(get_vector_weights(model))
    print('Individual size: {}'.format(weight_size))
    initial = np.random.rand(GEN_SIZE, weight_size).tolist()

    ga = GeneticAlgorithm(gen_size=GEN_SIZE, fitness=fitness_parallel,
                        gen_child=gen_child, best_to_keep=0.3,
                        mutation=None, mutation_probability=MUTATION_PROBABILITY,
                        random_gen=None)
    res, last_gen = ga.run(initial, generations=GEN_NUM, env_name='CartPole-v0', 
                           folder=folder, debug=False, plot=True, parallelize=True, 
                           callback=callback, n_jobs=N_JOBS)

    env = wrappers.Monitor(env, folder + 'video/', force=True)
    fit = fitness_parallel([res], env, render=True)[0]
    print('Final fitness: {}'.format(fit))

    # save model weights
    model.save_weights(folder + 'model_weights.weights')

    # save last generation
    save_generation(last_gen, folder + 'last_gen.csv')