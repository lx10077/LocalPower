from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np


filenames = [
    'a9a.txt', 'abalone.txt', 'acoustic.txt', 'aloi.txt', 'combined.txt',
    'connect-4.txt', 'covtype.txt', 'housing.txt', 'ijcnn1.txt', 'mnist.txt',
    'poker.txt', 'space_ga.txt', 'splice.txt', 'w8a.txt', 'YearPredictionMSD.txt'
]

np.random.seed(0)
for ind in range(len(filenames)):
    filename = filenames[ind]
    X, _ = load_svmlight_file('data/' + filename)
    X = np.array(X.todense())
    n, d = X.shape

    k = 4  # set the y axis
    m = 20
    s = math.ceil(n / m)

    perm = np.random.permutation(n)
    X = X[perm, :]
    # print('Size of X is ' + str(n) + '-by-' + str(d))
    # print('rank:', np.linalg.matrix_rank(X))

    # scale the feature to [-1, 1]
    X = MinMaxScaler().fit_transform(X)
    # print('X shape:', X.shape)

    M = X.transpose() @ X / n
    data = list()
    etas = []
    etas2 = []
    for i in range(m):
        idx_start = i * s
        idx_end = min((i+1)*s, n)
        Xi = X[idx_start:idx_end, :]
        ni = Xi.shape[0]

        data.append(Xi)
        Mi = Xi.transpose() @ Xi / ni

        etai = np.linalg.norm(Mi-M, 2)/sgm1
        etas.append(etai)
    print(filename, max(etas))




