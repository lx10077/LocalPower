from sklearn.datasets import load_svmlight_file
from pm import lpm_iden, lpm_sign, lpm_orth, DR_SVD, UDA, WDA
from sklearn.preprocessing import MinMaxScaler
import math
import numpy
import matplotlib.pyplot as plt


filenames = [
    'a9a.txt', 'abalone.txt', 'acoustic.txt', 'aloi.txt', 'combined.txt',
    'connect-4.txt', 'covtype.txt', 'housing.txt', 'ijcnn1.txt', 'mnist.txt',
    'poker.txt', 'space_ga.txt', 'splice.txt', 'w8a.txt', 'YearPredictionMSD.txt'
]

iterations = [
    30, 15, 30, 30, 40,
    40, 30, 15, 70, 30,
    20, 15, 30, 20, 20
]

decay = False

seeds = [13242, 22342, 45365, 36477, 54865, 63956, 78395, 89357, 90375, 103659]
for seed in seeds:
    print(seed, '================================')
    numpy.random.seed(seed)
    error_lst = []
    for ind in range(len(filenames)):
        filename = filenames[ind]
        X, _ = load_svmlight_file('data/' + filename)
        X = numpy.array(X.todense())
        n, d = X.shape

        k = 4  # set the y axis
        m = max(math.ceil(n/1000), 3)  # how many nodes
        t = iterations[ind]  # total steps
        s = math.ceil(n / m)

        perm = numpy.random.permutation(n)
        X = X[perm, :]
        print('Size of X is ' + str(n) + '-by-' + str(d))

        # scale the feature to [-1, 1]
        X = MinMaxScaler().fit_transform(X)
        print('X shape:', X.shape)

        u, sig, v = numpy.linalg.svd(X, full_matrices=False, compute_uv=True)
        v0 = v[0, :]
        vk = v[0:k+1, :]

        # Random initial
        q0 = numpy.random.randn(d, k + 1)
        q0, _ = numpy.linalg.qr(q0)

        data = list()
        for i in range(m):
            idx_start = i * s
            idx_end = min((i+1)*s, n)
            data.append(X[idx_start:idx_end, :])

        p = 4
        _, error_orth = lpm_orth(data, k, t, p, q=q0, vk=vk, is_decay=decay)
        _, error_sign = lpm_sign(data, k, t, p, q=q0, vk=vk, is_decay=decay)
        _, error_iden = lpm_iden(data, k, t, p, q=q0, vk=vk, is_decay=decay)

        iters = numpy.arange(t)
        fig = plt.figure(figsize=(8, 6))
        plt.semilogy(iters, error_orth, '-xg', label='orth', linewidth=4)
        plt.semilogy(iters, error_sign, '-or', label='sign', linewidth=4)
        plt.semilogy(iters, error_iden, '--*b', label='iden', linewidth=4)
        plt.xlabel('Communications', FontSize=28)
        plt.ylabel(r'$ sin \theta( Z_t , U_5) $', FontSize=30)
        plt.xticks([0, 5, 10, 15, 20], FontSize=26)
        plt.yticks(FontSize=26)
        plt.legend(fontsize=24)
        plt.tight_layout()
        plt.show()
        short_name = filename.split('.')[0]
        name = 'd_' + short_name + f"_p{p}_k{k+1}_m{m}.pdf"
        fig.savefig(name, format='pdf', dpi=1200)

        error1 = DR_SVD(X, k, iter=1, vk=vk)
        error2 = UDA(data, k, vk=vk)
        error3 = WDA(data, k, vk=vk)

        # errors = [error_orth[-1], error_sign[-1], error_iden[-1]]

        errors = [error_orth[-1], error_sign[-1], error_iden[-1],
                  error1, error2, error3
                  ]

        error_lst.append(errors)
        print(filename, errors)

        numpy.save(f'{seed}_decay', error_lst)
