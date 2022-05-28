import numpy as np

def incre_1(triplets, data_size):
    """ from a set of triplets, construct a distance matrix with arbitrary rules"""

    X = np.zeros([data_size,data_size])
    mask = np.zeros([data_size,data_size])
    np.fill_diagonal(mask, 1)

    for a,p,n in triplets:
        X[a,p] -= 1
        X[a,n] += 1
        X[p,a] -= 1
        X[n,a] += 1
        mask[a,p], mask[a,n], mask[p,a], mask[n,a] = 1, 1, 1, 1

    np.fill_diagonal(X, X.min()*2)

    return X, mask

def incre_rand(triplets, data_size):
    """ from a set of triplets, construct a distance matrix with arbitrary rules"""

    X = np.zeros([data_size,data_size])
    mask = np.zeros([data_size,data_size])
    np.fill_diagonal(mask, 1)

    for a,p,n in triplets:
        X[a,p] -= np.random.random()
        X[a,n] += np.random.random()
        X[p,a] -= np.random.random()
        X[n,a] += np.random.random()
        mask[a,p], mask[a,n], mask[p,a], mask[n,a] = 1, 1, 1, 1

    np.fill_diagonal(X, X.min()*2)

    return X, mask

def incre_pos(triplets, data_size):
    """ from a set of triplets, construct a distance matrix with arbitrary rules"""
    X = np.zeros([data_size,data_size])
    mask = np.zeros([data_size,data_size])
    np.fill_diagonal(mask, 1)

    for a,p,n in triplets:
        X[a,p] += 1
        X[a,n] += 2
        X[p,a] += 1
        X[n,a] += 2
        mask[a,p], mask[a,n], mask[p,a], mask[n,a] = 1, 1, 1, 1

    np.fill_diagonal(X, 0)

    return X, mask
