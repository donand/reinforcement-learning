import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool

import gym

class GeneticAlgorithm:
    # fitness - callable, computes fitness value
    def __init__(self, gen_size, fitness, gen_child, best_to_keep, mutation, mutation_probability, random_gen):
        self.gen_size = gen_size
        self.fitness = fitness
        self.gen_child = gen_child
        self.best_to_keep = int(best_to_keep * gen_size)
        self.mutation = mutation
        self.mutation_probability = mutation_probability
        self.random_gen = random_gen

    def compute_fitness(self, curr_gen, env_name, n_jobs, parallelize):
        if parallelize:
            splits = []
            envs = [gym.make(env_name).env for i in range(n_jobs)]
            interval = len(curr_gen) // n_jobs
            for i in range(n_jobs):
                i2 = len(curr_gen) if i == n_jobs - 1 else (i + 1) * interval
                splits.append((curr_gen[i * interval: i2], envs[i]))
            with Pool(n_jobs) as p:
                res = p.starmap(self.fitness, splits)
            tmp = []
            for i in range(n_jobs):
                tmp += res[i]
            fitness = [(i, tmp[i]) for i in range(self.gen_size)]
        else:
            fitness = [(i, self.fitness(curr_gen[i], envs[0])) for i in range(self.gen_size)]
        return fitness

    def run(self, first_generation, generations, env_name, folder, plot=False, callback=None, debug=False, parallelize=True, n_jobs=1):
        curr_gen = first_generation
        best_fitnesses = []
        fitnesses = []
        for gen in range(generations):
            # evaluate fitness (parallelizable)
            fitness = self.compute_fitness(curr_gen, env_name, n_jobs, parallelize)
            # keep best
            fitness_sorted = sorted(fitness, key=lambda x: x[1], reverse=True)
            if gen % 10 == 0:
                print('Generation {}, fitness: {}'.format(gen, fitness_sorted[0][1]))
            best_fitnesses.append(fitness_sorted[0][1])
            best = fitness_sorted[:self.best_to_keep]
            best = [curr_gen[i[0]] for i in best]
            # breed
            fitness_norm = np.array([x[1] for x in fitness])
            fitnesses.append(fitness_norm)
            fitness_norm = fitness_norm / np.sum(fitness_norm)
            children = []
            if debug:
                print()
                print('Fitness: {}'.format(fitness))
                print()
                print('Fitness_sorted: {}'.format(fitness_sorted))
                print()
                print('Fitness_norm: {}'.format(fitness_norm))
            for i in range(self.best_to_keep, self.gen_size):
                # select parents
                #print(curr_gen)
                parents = np.random.choice(range(self.gen_size), p=fitness_norm, replace=False, size=2)
                if debug:
                    print()
                    #print('Parents: {}'.format(parents))
                # generate child
                children.append(self.gen_child((curr_gen[parents[0]], curr_gen[parents[1]])))
                if debug:
                    print()
                    #print('New child: {}'.format(children[-1]))
            # create new generation
            curr_gen = best + children
            # plot
            if plot:
                pass
            # callback
            if callback:
                callback(gen, curr_gen, best_fitnesses, fitnesses)
        if plot:
            fig = plt.figure(figsize=(16,8))
            plt.plot(best_fitnesses)
            plt.savefig(folder + 'fitness_per_generation.svg', format='svg')
            plt.show()
            fig = plt.figure(figsize=(16,8))
            plt.boxplot(fitnesses)
            plt.savefig(folder + 'fitness_per_generation_boxplot.svg', format='svg')
            plt.show()

        fitness = self.compute_fitness(curr_gen, env_name, n_jobs, parallelize)
        fitness_sorted = sorted(fitness, key=lambda x: x[1], reverse=True)
        return curr_gen[fitness_sorted[0][0]], curr_gen