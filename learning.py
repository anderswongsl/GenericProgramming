import numpy as np
def get_data(file):
    train = np.genfromtxt(file,delimiter=",")
    real_results = np.empty(train.shape[0])
    for n in range(train.shape[0]):
        real_results[n] = train[n][-1]
        train[n][-1] = 1
    return train, real_results

def training(train,real_results,copy_rate):
    weights = np.random.uniform(-1,1,train.shape[1])
    fitness = 0
    while(fitness < train.shape[0]-1):
        fitness = 0
        for n in range(train.shape[0]):
            result = np.dot(weights[:-1],train[n][:-1])> -weights[-1]
            if result == real_results[n]:
                fitness+=1
            else:
                weights = weights + copy_rate*(real_results[n]-result)*train[n]
    return weights