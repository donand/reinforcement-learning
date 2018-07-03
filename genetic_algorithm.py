import matplotlib.pyplot as plt
import numpy as np

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

    def run(self, first_generation, generations, plot=False, callback=None, debug=False):
        curr_gen = first_generation
        best_fitnesses = []
        for gen in range(generations):
            # evaluate fitness (parallelizable)
            fitness = [(i, self.fitness(curr_gen[i])) for i in range(self.gen_size)]
            # keep best
            fitness_sorted = sorted(fitness, key=lambda x: x[1], reverse=True)
            if gen % 10 == 0:
                print('Generation {}, fitness: {}'.format(gen, fitness_sorted[0][1]))
            best_fitnesses.append(fitness_sorted[0][1])
            best = fitness_sorted[:self.best_to_keep]
            best = [curr_gen[i[0]] for i in best]
            # breed
            fitness_norm = np.array([x[1] for x in fitness])
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
                    print('Parents: {}'.format(parents))
                # mutation
                # generate child
                children.append(self.gen_child((curr_gen[parents[0]], curr_gen[parents[1]])))
                if debug:
                    print()
                    print('New child: {}'.format(children[-1]))
            # create new generation
            curr_gen = best + children
            # plot
            if plot:
                pass
            # callback
            if callback:
                pass
        if plot:
            fig = plt.figure(figsize=(16,8))
            plt.plot(best_fitnesses)
            plt.savefig('fitness_per_generation.svg', format='svg')
            plt.show()
        fitness = [(i, self.fitness(curr_gen[i])) for i in range(len(curr_gen))]
        fitness_sorted = sorted(fitness, key=lambda x: x[1], reverse=True)
        return curr_gen[fitness_sorted[0][0]]