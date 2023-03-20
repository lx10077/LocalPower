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
    40, 30, 15, 90, 30,
    20, 15, 30, 20, 20
]

iterations0 = [
    20, 10, 25, 15, 40,
    40, 20, 15, 70, 20,
    20, 10, 15, 20, 20
]


numpy.random.seed(000)
from_data = False
decay = True

method = 'iden'
if method == 'orth':
    lpm = lpm_orth
elif method == 'sign':
    lpm = lpm_sign
else:
    lpm = lpm_iden

if not from_data:
    result = dict()
else:
    result = numpy.load('p_sign.npy', allow_pickle=True).tolist()


fig, axs = plt.subplots(5, 3, figsize=(10, 13))
for row in range(5):
    for col in range(3):
        ax_i = axs[row, col]
        ind = 3 * row + col
        filename = filenames[ind]
        t = iterations[ind]  # total steps

        if not from_data:
            X, _ = load_svmlight_file('data/' + filename)
            X = numpy.array(X.todense())
            n, d = X.shape

            k = 4  # set the y axis
            m = max(math.ceil(n/1000), 3)  # how many nodes
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
            t = iterations[ind]
            _, error1 = lpm(data, k, t, 1, q=q0, vk=vk, is_decay=False)
            _, error2 = lpm(data, k, t, 2, q=q0, vk=vk, is_decay=decay, exp=True)
            _, error4 = lpm(data, k, t, 4, q=q0, vk=vk, is_decay=decay, exp=True)
            _, error8 = lpm(data, k, t, 8, q=q0, vk=vk, is_decay=decay, exp=True)
            _, error16 = lpm(data, k, t, 16, q=q0, vk=vk, is_decay=decay, exp=True)
            _, error32 = lpm(data, k, t, 32, q=q0, vk=vk, is_decay=decay, exp=True)

            result[filename] = [
                error1, error2, error4, error8, error16, error32
            ]
            numpy.save(f'p_{method}_decay', result)

        else:
            error1, error2, error4, error8, error16 = result[filename]

        iters = numpy.arange(t)
        ax_i.semilogy(iters, error1[:t], '-xg', label='p=1', linewidth=1)
        ax_i.semilogy(iters, error2[:t], '--*b', label='p=2', linewidth=1)
        ax_i.semilogy(iters, error4[:t], '-*c', label='p=4', linewidth=1)
        ax_i.semilogy(iters, error8[:t], '-or', label='p=8', linewidth=1)
        ax_i.semilogy(iters, error16[:t], '-.m', label='p=16', linewidth=1)
        ax_i.semilogy(iters, error32[:t], ':s', label='p=32', linewidth=1)

        title_i = filenames[ind].split(".")[0]
        ax_i.set_title(title_i)


plt.legend(fontsize=10)
plt.xlabel('Communications', FontSize=10)
plt.ylabel(r'$ sin \theta( Z_t , U_5) $', FontSize=10)
plt.tight_layout()
name = f"p_{method}_decay.pdf"
fig.savefig(name, format='pdf', dpi=1200)
