import numpy as np
import math
train = np.genfromtxt("gp-training-set.csv",delimiter= ",")
weight_size = train.shape[1] - 1
gen_size = 10000
copy_size = 1000
weight_limit = 2
threshold_limit = 3
crossover_size = gen_size - copy_size
mutation_size = 10
def initialize(gen_size):
    generation = [np.random.uniform(-weight_limit,weight_limit,weight_size+1) for n in range(gen_size)]
    for n in range(gen_size):
        generation[n][weight_size] = np.random.uniform(-threshold_limit,threshold_limit)
    return np.array(generation)

def score_cal(weights,train):
    output = np.array([1 if np.dot(weights[:weight_size],train[i][:weight_size])>=weights[weight_size] else 0 for i in range(train.shape[0])])
    real = np.array([train[i][weight_size] for i in range(train.shape[0])])
    score = np.sum(output == real)
    return score

def fitness(generation,train):
    arr = np.array([score_cal(generation[n],train) for n in range(generation.shape[0])])
    return arr

def copy(generation,fitness,copy_size):
    copy_index = np.argpartition(-fitness,copy_size)[:copy_size]
    new_generation = generation[copy_index]
    return new_generation

def mutation(generation,mutation_size):
    for n in range(mutation_size):
        index = np.random.randint(0,generation.shape[0]-1)
        position = np.random.randint(0,weight_size)
        generation[index][position:-1] = np.random.uniform(-weight_limit,weight_limit,generation[index][position:-1].shape[0])
        generation[index][weight_size] = np.random.uniform(-threshold_limit,threshold_limit)
    return

def crossover(generation,fitness,crossover_size,each_round = 10,):
    result = np.empty((crossover_size,weight_size+1))
    for n in range(0,crossover_size,2):
        index = np.argpartition(-np.random.choice(fitness, size=each_round, replace=False), 2)[:2]
        position = np.random.randint(0, weight_size - 1)
        generation[[0, 1]], generation[index] = generation[index], generation[[0, 1]]
        result[n] = np.concatenate([generation[0][:position], generation[1][position:]])
        result[n+1] = np.concatenate([generation[1][:position], generation[0][position:]])
        generation = generation[2:]
    return result

generation = initialize(gen_size)
fitness_max = 0
while(fitness_max !=100):
    new_gen = copy(generation,fitness(generation,train),copy_size)
    mutation(generation,mutation_size)
    generation = np.concatenate([crossover(generation,fitness(generation,train),crossover_size,10),new_gen])
    fitness_scores = fitness(generation,train)
    fitness_max = fitness_scores.max()
    print(fitness_max)
    print(generation[np.argmax(fitness_scores)])
print(generation[np.argmax(fitness_scores)])