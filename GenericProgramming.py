import numpy as np
import math
train = np.genfromtxt("gp-training-set.csv",delimiter= ",")
weight_size = train.shape[1] - 1
gen_size = 1000
copy_size = 100
weight_limit = 5
threshold_limit = 10
crossover_size = gen_size - copy_size
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
    generation = np.delete(generation,copy_index,0)
    return generation, new_generation

def mutation(generation,mutation_size):
    for n in range(mutation_size):
        index = np.random.randint(0,generation.shape[0])
        position = np.random.randint(0,weight_size)
        dir = np.random.choice([True,False])
        if dir:
            generation[index][position:-1] = np.random.uniform(-weight_limit,weight_limit,generation[index][position:-1].shape[0])
        else:
            generation[index][:position] = np.random.uniform(-weight_limit,weight_limit,generation[index][:position].shape[0])
        generation[index][weight_size] = np.random.uniform(-threshold_limit,threshold_limit)
    return
def crossover_each(generation,fitness,each_round):
    each_round = each_round if each_round<generation.shape[0] else generation.shape[0]
    index = np.argpartition(-np.random.choice(fitness,size = each_round,replace = False),2)[:2] if generation.shape[0]>2 else [0,1]
    position = np.random.randint(0,weight_size-1)
    generation[[0,1]],generation[index] = generation[index],generation[[0,1]]
    fitness[[0, 1]], fitness[index] = fitness[index], fitness[[0, 1]]
    new1 = np.concatenate([generation[0][:position],generation[1][position:]])
    new2 = np.concatenate([generation[1][:position],generation[0][position:]])
    return new1,new2,generation[2:],fitness[2:]

def crossover(generation,fitness,crossover_size,each_round = 10,):
    result = np.empty((crossover_size,weight_size+1))
    for n in range(0,crossover_size,2):
        result[n],result[n+1],generation,fitness = crossover_each(generation,fitness,each_round)
    return result

gen = initialize(gen_size)
fitness_max = 0
while(fitness_max !=100):
    fitness_scores = fitness(gen,train)
    gen,new_gen = copy(gen,fitness_scores,copy_size)
    mutation(gen,1)
    fitness_scores = fitness(gen,train)
    gen = np.concatenate([crossover(gen,fitness_scores,crossover_size,10),new_gen])
    fitness_scores = fitness(gen,train)
    fitness_max = fitness_scores.max()
    print(fitness_max)
    print(gen[np.argmax(fitness_scores)])
print(gen[np.argmax(fitness_scores)])

