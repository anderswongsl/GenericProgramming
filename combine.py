import numpy as np
import math
train = np.genfromtxt("gp-training-set.csv",delimiter= ",")
weight_size = train.shape[1] - 1
gen_size = 10000
copy_size = 1000
weight_limit = 2
threshold_limit = 3
crossover_size = gen_size - copy_size
mutation_size = 100
def initialize(generation_size):
    gen = np.empty((generation_size, weight_size + 1))
    for i in range(generation_size):
        perceptron = np.random.uniform(-weight_limit, weight_limit, weight_size + 1)
        perceptron[weight_size] = np.random.uniform(-threshold_limit, threshold_limit)
        gen[i] = perceptron

    return gen


def fitness_cal(perceptron,train):
    score = 0
    for i in range(train.shape[0]):
        sum = np.dot(train[i][:weight_size], perceptron[:weight_size])
        output = 1 if sum >= perceptron[weight_size] else 0
        if output == train[i][weight_size]:
            score += 1

    return score


def fitness(gen,train):
    scores = np.empty(gen.shape[0])
    for i in range(gen.shape[0]):
        scores[i] = fitness_cal(gen[i],train)

    return scores

def copy(old_gen, fitness_arr,copy_size):
    # np.random.shuffle(old_gen)
    copied = np.empty((copy_size, weight_size + 1))
    # for i in range(copy_size):
    #     index_to_copy = np.argmax(np.random.choice(fitness_arr, size=tournament_size, replace=False))
    #     old_gen[0], old_gen[index_to_copy] = old_gen[index_to_copy], old_gen[0]
    #     copied[i] = old_gen[0]
    #     old_gen = old_gen[1:]
    indices_to_copy = np.argpartition(-fitness_arr, copy_size)[:copy_size]
    copied = old_gen[indices_to_copy]
    # old_gen = np.take(old_gen, np.setdiff1d(np.arange(old_gen.shape[0] - indices_to_copy)))
    old_gen = np.delete(old_gen, indices_to_copy)

    return copied


def mutation(old_gen,mutate_size):
    for i in range(mutate_size):
        mutate_index = np.random.randint(0, old_gen.shape[0] - 1)
        pos = np.random.randint(0, weight_size - 1)
        old_gen[mutate_index][pos:-1] = np.random.uniform(-weight_limit, weight_limit,
                                                          old_gen[mutate_index][pos:-1].shape[0])
        old_gen[mutate_index][-1] = np.random.uniform(-threshold_limit, threshold_limit)

def crossover(old_gen, fitness ,crossover_size,each_round):
    crossovered = np.empty((crossover_size, weight_size + 1))
    for i in range(0, crossover_size, 2):
        indices_parents = np.argpartition(-np.random.choice(fitness, size=each_round, replace=False), 2)[:2]
        pos = np.random.randint(1, weight_size)
        old_gen[[0, 1]], old_gen[indices_parents] = old_gen[indices_parents], old_gen[[0, 1]]
        crossovered[i] = np.concatenate([old_gen[0][:pos], old_gen[1][pos:]])
        crossovered[i + 1] = np.concatenate([old_gen[1][:pos], old_gen[0][pos:]])
        old_gen = old_gen[2:]

    return crossovered

gen = initialize(gen_size)
fitness_max = 0
fitness_scores = fitness(gen,train)
while(fitness_max !=100):
    new_gen = copy(gen,fitness(gen,train),copy_size)
    mutation(gen,mutation_size)
    gen = np.concatenate([new_gen,crossover(gen,fitness(gen,train),crossover_size,10)])
    fitness_scores = fitness(gen,train)
    fitness_max = fitness_scores.max()
    print(fitness_max)
    print(gen[np.argmax(fitness_scores)])
print(gen[np.argmax(fitness_scores)])
