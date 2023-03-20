from sklearn.datasets import load_svmlight_file
from pm import lpm_orth, lpm_sign, lpm_iden
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
    30, 20, 30, 30, 40,
    40, 30, 20, 70, 30,
    20, 20, 30, 20, 20
]
ms = [20, 40, 60, 80]

seed = 5555
numpy.random.seed(seed)
from_data = False

fig, axs = plt.subplots(5, 3, figsize=(10, 13))

method = 'orth'
lpm = lpm_orth
p = 4
k = 4  # set the y axis

if not from_data:
    result = dict()
else:
    result = numpy.load(f'm_{method}_{seed}.npy', allow_pickle=True).tolist()


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

            result_m = []
            for m in ms:
                s = math.ceil(n / m)
                data = list()
                for i in range(m):
                    idx_start = i * s
                    idx_end = min((i+1)*s, n)
                    data.append(X[idx_start:idx_end, :])

                t = iterations[ind]
                _, error = lpm(data, k, t, p, q=q0, vk=vk, is_decay=False)
                result_m.append(error)

            result[filename] = result_m
            numpy.save(f'm_{method}_{seed}', result)

        error_m20, error_m40, error_m60, error_m80 = result[filename]
        iters = numpy.arange(t)
        ax_i.semilogy(iters, error_m80[:t], '-xg', label='m=80', linewidth=1)
        ax_i.semilogy(iters, error_m60[:t], '--*b', label='m=60', linewidth=1)
        ax_i.semilogy(iters, error_m40[:t], '-*c', label='m=40', linewidth=1)
        ax_i.semilogy(iters, error_m20[:t], '-or', label='m=20', linewidth=1)

        title_i = filenames[ind].split(".")[0]
        ax_i.set_title(title_i)


plt.legend(fontsize=10)
plt.xlabel('Communications', FontSize=10)
plt.ylabel(r'$ sin \theta( Z_t , U_5) $', FontSize=10)
plt.tight_layout()
name = f"m_{method}_{seed}.pdf"
fig.savefig(name, format='pdf', dpi=1200)
