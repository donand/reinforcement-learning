# Reinforcement Learning Experiments
In this repository I collect some experiments of different techniques to solve some reinforcement learning environments provided by OpenAI.

## Genetic Algorithms
I wrote an implementation of genetic algorithms that can be found in the `genetic_algorithm.py` file.

The algorithm lets the user define its own functions for the computation of the fitness and the generation of a child given two parents. <br>
The representation of each individual of a generation is up to the user. <br>
The top k% of the individuals are selected to be kept in the next generation. <br>
The selection of the parents is performed by sampling the generation weighting the individuals on their fitness value. So individuals with bigger fitness value are more likely to be selected as a parent for the new offspring.

The implementation supports multiprocessing to compute in parallel the fitness of one generation. In this way I can split the computation into N different processes.

## OpenAI Gym CartPole using Genetic Algorithms
I am using my implementation of genetic algorithms in order to train a neural network that is used as the policy by the agent.

The environment that I am going to solve is the [CartPole-v0](https://github.com/openai/gym/wiki/CartPole-v0) by OpenAI. 

One generation is made of several individuals, and each individual is a configuration of the neural network. <br>
One configuration is represented with a flattened array of all the weights of the network. <br>
The fitness value of one configuration is computed by running one episode of the environment and returning the sum of the rewards. <br>
The generation of a child is performed by a single point crossover, with the point chosen randomly inside the array of the weights. After this, I add to each weight a random sample taken from a normal distribution with a probability given by the mutation probability.

The parameters of the genetic algorithms are the following:

| Parameter                               | Value |
| --------------------------------------- | ----- |
| Generation Size                         | 40    |
| Number of Generations                   | 50    |
| Max number of steps for the environment | 1000  |
| Mutation Probability                    | 0.05  |


The structure of the network is the following:

| Layer         | Size | Activation |
| ------------- | ---- | ---------- |
| Input         | 4    |
| First hidden  | 10   | ReLU       |
| Second hidden | 7    | ReLU       |
| Output        | 1    | Sigmoid    |

The input of the network will be the observation provided by the environment, that is composed by 4 real values. The output will be a discrete binary action {Left, Right}.

The fitness during the 50 generations is

![Fitness](cartpole_ga/fitness_per_generation.svg)

At the end, the agent learnt how to keep the pole balanced.

<p align="center">
<img src="cartpole_ga/video/cartpole.gif">
</p>


## OpenAI Gym MountainCar using Genetic Algorithms
The environment that I am going to solve is the [MountainCar-v0](https://github.com/openai/gym/wiki/MountainCar-v0) by OpenAI. 

We have the same settings of the problem with the CartPole-v0, we have generations of networks represented by the weights. <br>
The crossover and mutation operators are the same.

The differences are in the network structure and in the parameters of the genetic algorithm.

The parameters of the genetic algorithms are the following:

| Parameter                               | Value |
| --------------------------------------- | ----- |
| Generation Size                         | 50    |
| Number of Generations                   | 100   |
| Max number of steps for the environment | 1000  |
| Mutation Probability                    | 0.1   |

The structure of the network is the following:

| Layer         | Size | Activation |
| ------------- | ---- | ---------- |
| Input         | 2    |
| First hidden  | 10   | ReLU       |
| Second hidden | 7    | ReLU       |
| Output        | 3    | Softmax    |

The output layer has one neuron for each action.

The input to the network is the observation provided by the environment, that are the position and the speed of the car. The output of the network will be one discrete action in {Left, Nothing, Right}.

The fitness values for different generations are the following

|   Generation   |                                       Plot                                       |                                         Boxplot                                          |
| :------------: | :------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------: |
| Generation 20  | ![](mountaincar_ga/2018-07-05_11-24/checkpoints/generation_20/generation_20.svg) | ![](mountaincar_ga/2018-07-05_11-24/checkpoints/generation_20/generation_20_boxplot.svg) |
| Generation 40  | ![](mountaincar_ga/2018-07-05_11-24/checkpoints/generation_40/generation_40.svg) | ![](mountaincar_ga/2018-07-05_11-24/checkpoints/generation_40/generation_40_boxplot.svg) |
| Generation 60  | ![](mountaincar_ga/2018-07-05_11-24/checkpoints/generation_60/generation_60.svg) | ![](mountaincar_ga/2018-07-05_11-24/checkpoints/generation_60/generation_60_boxplot.svg) |
| Generation 80  | ![](mountaincar_ga/2018-07-05_11-24/checkpoints/generation_80/generation_80.svg) | ![](mountaincar_ga/2018-07-05_11-24/checkpoints/generation_80/generation_80_boxplot.svg) |
| Generation 100 |         ![](mountaincar_ga/2018-07-05_11-24/fitness_per_generation.svg)          |         ![](mountaincar_ga/2018-07-05_11-24/fitness_per_generation_boxplot.svg)          |

The resulting agents in different generations are the following:

 |                                                                                                                                 |                                                                                                                                  |
 | :-----------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------: |
 |  <img src="mountaincar_ga/2018-07-05_11-24/checkpoints/generation_0/openaigym.video.0.8543.video000000.gif"> <br> Generation 1  | <img src="mountaincar_ga/2018-07-05_11-24/checkpoints/generation_20/openaigym.video.2.8543.video000000.gif"> <br> Generation 20  |
 | <img src="mountaincar_ga/2018-07-05_11-24/checkpoints/generation_40/openaigym.video.4.8543.video000000.gif"> <br> Generation 40 | <img src="mountaincar_ga/2018-07-05_11-24/checkpoints/generation_60/openaigym.video.6.8543.video000000.gif">  <br> Generation 60 |
 | <img src="mountaincar_ga/2018-07-05_11-24/checkpoints/generation_80/openaigym.video.8.8543.video000000.gif"> <br> Generation 80 |          <img src="mountaincar_ga/2018-07-05_11-24/video/openaigym.video.10.8543.video000000.gif"> <br> Generation 100           |
