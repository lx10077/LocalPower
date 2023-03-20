from sklearn.datasets import load_svmlight_file
from pm import lpm_iden, lpm_sign, lpm_orth
from sklearn.preprocessing import MinMaxScaler
import math
import numpy
import matplotlib.pyplot as plt



filenames = [
    'a9a.txt', 'abalone.txt', 'acoustic.txt', 'aloi.txt', 'combined.txt',
    'connect-4.txt', 'covtype.txt', 'housing.txt', 'ijcnn1.txt', 'mnist.txt',
    'poker.txt', 'space_ga.txt', 'splice.txt', 'w8a.txt', 'YearPredictionMSD.txt'
]
p_max = 20
k = 4  # set the y axis

ms = [20, 30, 40, 50]
seeds = [152]

iterations = [
    30, 15, 30, 30, 40,
    40, 30, 15, 70, 30,
    20, 15, 30, 20, 20
]
method = 'orth'

result_dict = dict()
for ind in range(len(filenames)):
    filename = filenames[ind]
    print(ind, filename, '=====================================================================')
    X, _ = load_svmlight_file('data/' + filename)
    X = numpy.array(X.todense())
    n, d = X.shape

    X = MinMaxScaler().fit_transform(X)  # scale the feature to [-1, 1]
    u, sig, v = numpy.linalg.svd(X, full_matrices=False, compute_uv=True)
    v0 = v[0, :]
    vk = v[0:k + 1, :]
    t = iterations[ind]

    seed_lst = []
    for seed in seeds:
        print(seed)
        numpy.random.seed(seed)
        perm = numpy.random.permutation(n)
        X = X[perm, :]

        q0 = numpy.random.randn(d, k + 1)
        q0, _ = numpy.linalg.qr(q0)

        errors_m = []
        for m in ms:
            s = math.ceil(n / m)

            data = list()
            for i in range(m):
                idx_start = i * s
                idx_end = min((i + 1) * s, n)
                data.append(X[idx_start:idx_end, :])

            errors = []
            for p in range(1, p_max+1):
                _, error = lpm_orth(data, k, t, p, q=q0, vk=vk, is_decay=False)
                errors.append(error[-1])

            errors_m.append(errors)
        seed_lst.append(errors_m)
    result_dict[filename] = seed_lst
    numpy.save(f'sin_m_p_{method}', result_dict)



result = numpy.load(f'sin_m_p_{method}.npy', allow_pickle=True).tolist()
fig, axs = plt.subplots(5, 3, figsize=(10, 13))

for row in range(5):
    for col in range(3):
        ax_i = axs[row, col]
        ind = 3 * row + col
        filename = filenames[ind]
        seed_lst = result[filename]
        repeated_result = numpy.array(seed_lst)

        mean_result = repeated_result.mean(0)
        std = repeated_result.std(0)
        m20 = mean_result[0]
        m40 = mean_result[1]
        m60 = mean_result[2]
        m80 = mean_result[3]

        iters = numpy.arange(p_max)
        ax_i.semilogy(iters, m80[:t], '-xg', label='m=8', linewidth=1)
        ax_i.semilogy(iters, m60[:t], '--*b', label='m=6', linewidth=1)
        ax_i.semilogy(iters, m40[:t], '-*c', label='m=4', linewidth=1)
        ax_i.semilogy(iters, m20[:t], '-or', label='m=2', linewidth=1)

        title_i = filenames[ind].split(".")[0]
        ax_i.set_title(title_i)

plt.legend(fontsize=10)
plt.xlabel('p', FontSize=10)
plt.ylabel(r'$ sin \theta( Z_t , U_5) $', FontSize=10)
plt.tight_layout()
fig.savefig(f'sin_m_p_{method}_new1.pdf', format='pdf', dpi=1200)
