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
- Size of a generation: 40
- Number of generations: 50
- Max number of steps for the environment: 1000
- Mutation probability: 0.05

The structure of the network is the following:
- Input layer size: 4
- First hidden layer: 10 neurons with ReLU activation function
- Second hidden layer: 7 neurons with ReLU activation funtion
- Output layer: 1 neuron with sigmoid activation function

The input of the network will be the observation provided by the environment, that is composed by 4 real values. The output will be a discrete binary action {Left, Right}.

The fitness during the 50 generations is

![Fitness](cartpole_ga/fitness_per_generation.svg)

At the end, the agent learnt how to keep the pole balanced.

<p align="center">
<img src="https://media.giphy.com/media/3mjTMF04baZzRWwCK8/giphy.gif">
</p>


## OpenAI Gym MountainCar using Genetic Algorithms
The environment that I am going to solve is the [MountainCar-v0](https://github.com/openai/gym/wiki/MountainCar-v0) by OpenAI. 

We have the same settings of the problem with the CartPole-v0, we have generations of networks represented by the weights. <br>
The crossover and mutation operators are the same.

The differences are in the network structure and in the parameters of the genetic algorithm.

The parameters of the genetic algorithms are the following:
- Size of a generation: 50
- Number of generations: 100
- Max number of steps for the environment: 1000
- Mutation probability: 0.1

The structure of the network is the following:
- Input layer size: 2
- First hidden layer: 10 neurons with ReLU activation function
- Second hidden layer: 7 neurons with ReLU activation funtion
- Output layer: 3 neurons with softmax activation function, one for each discrete action

The input to the network is the observation provided by the environment, that are the position and the speed of the car. The output of the network will be one discrete action in {Left, Nothing, Right}.

The fitness values for different generations are the following

Generation               |  Plot             |  Boxplot
:-------------------------:|:-------------------------:|:-------------------------:
Generation 20  |  ![](mountaincar_ga/2018-07-05_11-24/checkpoints/generation_20/generation_20.svg) |  ![](mountaincar_ga/2018-07-05_11-24/checkpoints/generation_20/generation_20_boxplot.svg)
Generation 40  |  ![](mountaincar_ga/2018-07-05_11-24/checkpoints/generation_40/generation_40.svg) |  ![](mountaincar_ga/2018-07-05_11-24/checkpoints/generation_40/generation_40_boxplot.svg)
Generation 60  |  ![](mountaincar_ga/2018-07-05_11-24/checkpoints/generation_60/generation_60.svg) |  ![](mountaincar_ga/2018-07-05_11-24/checkpoints/generation_60/generation_60_boxplot.svg)
Generation 80  |  ![](mountaincar_ga/2018-07-05_11-24/checkpoints/generation_80/generation_80.svg) |  ![](mountaincar_ga/2018-07-05_11-24/checkpoints/generation_80/generation_80_boxplot.svg)
Generation 100  |  ![](mountaincar_ga/2018-07-05_11-24/fitness_per_generation.svg) |  ![](mountaincar_ga/2018-07-05_11-24/fitness_per_generation_boxplot.svg)

The resulting agents in different generations are the following:

 |     |     |
 | :---: | :---: |
<img src="https://media.giphy.com/media/3bb2y6HtMvGLWqsNMx/giphy.gif"> <br> Generation 1 | <img src="https://media.giphy.com/media/2YozJ6bVOkIaeEYZwP/giphy.gif"> <br> Generation 20 
<img src="https://media.giphy.com/media/1qfepCQhoct08oP6lM/giphy.gif"> <br> Generation 40  | <img src="https://media.giphy.com/media/claE43YJYmSW65hPkg/giphy.gif">  <br> Generation 60 
<img src="https://media.giphy.com/media/dCB33uxdq5O9l6M3Iw/giphy.gif"> <br> Generation 80  | <img src="https://media.giphy.com/media/8JNBtsnx7fK2w4Vrbp/giphy.gif"> <br> Generation 100 

