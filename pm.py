import numpy as np


def local_power(a, k, t, q=None, vk=None):
    n, d = a.shape
    if q is None:
        q = np.random.randn(d, k + 1)
        q, _ = np.linalg.qr(q)

    errors = np.zeros(t)
    aTa = np.dot(a.T, a)
    for j in range(t):
        sine = np.linalg.norm(np.matmul(np.identity(d)-np.matmul(vk.transpose(), vk), q[:, 0:k+1]), ord=2)
        errors[j] = sine

        z = np.dot(aTa, q)

        if j != t-1:
            q, _ = np.linalg.qr(z)

        if j == t-1:
            out = z
    return out, q, errors


def lpm_iden(data, k, t, p, q=None, vk=None, is_decay=False, exp=True):
    """
    Args:
        data: a list of m matrices, each of which is s-by-d
        k: target dimension
        t: number of outer loop iterations
        p: number of inner loop iterations
        q: d-by-k orthogonal basis
        vk: d-by-k target eigenvector matrix
        is_decay: whether to decay p
        exp: whether to decay p exponentially
    """
    m = len(data)
    d = data[0].shape[1]
    if q is None:
        q = np.random.randn(d, k + 1)
        q, _ = np.linalg.qr(q)
    errors = np.zeros(t)

    for j in range(t):
        sine = np.linalg.norm(np.matmul(np.identity(d)-np.matmul(vk.transpose(), vk), q[:, 0:k]), ord=2)
        # sine = np.dot(q[:, k], vk)
        errors[j] = sine

        z = np.zeros((d, k + 1))
        for i in range(m):
            qi, zi, _ = local_power(data[i], k, p, q=q, vk=vk)
            z += qi
        q, _ = np.linalg.qr(z)

        if is_decay:
            if exp:
                p = max(1, p//2)
            else:
                p = max(1, p - 1)
    return q, errors


def lpm_sign(data, k, t, p, q=None, vk=None, is_decay=False, exp=True):
    """
    Args:
        data: a list of m matrices, each of which is s-by-d
        k: target dimension
        t: number of outer loop iterations
        p: number of inner loop iterations
        q: d-by-k orthogonal basis
        vk: d-by-k target eigenvector matrix
        is_decay: whether to decay p
        exp: whether to decay p exponentially
    """
    m = len(data)
    d = data[0].shape[1]
    if q is None:
        q = np.random.randn(d, k + 1)
        q, _ = np.linalg.qr(q)
    errors = np.zeros(t)

    for j in range(t):
        sine = np.linalg.norm(np.matmul(np.identity(d) - np.matmul(vk.transpose(), vk), q[:, 0:k]), ord=2)
        # sine = np.dot(q[:, k], vk)
        errors[j] = sine

        z = np.zeros((d, k + 1))
        for i in range(m):
            qi, zi, _ = local_power(data[i], k, p, q=q, vk=vk)
            if i == 0:
                z0 = zi
            for ii in range(k + 1):
                if p != 1:
                    qi[:, ii] = qi[:, ii] * np.sign(np.matmul(z0[:, ii].T, zi[:, ii]))
            z += qi
        q, _ = np.linalg.qr(z)

        if is_decay:
            if exp:
                p = max(1, p//2)
            else:
                p = max(1, p - 1)
    return q, errors


def procrustes(u, u_base):
    U, D, V = np.linalg.svd(np.matmul(u.transpose(), u_base))
    P = np.matmul(U, V)
    return P


def lpm_orth(data, k, t, p, q=None, vk=None, is_decay=False, exp=True):
    """
    Args:
        data: a list of m matrices, each of which is s-by-d
        k: target dimension
        t: number of outer loop iterations
        p: number of inner loop iterations
        q: d-by-k orthogonal basis
        vk: d-by-k target eigenvector matrix
        is_decay: whether to decay p
        exp: whether to decay p exponentially
    """
    m = len(data)
    d = data[0].shape[1]
    if q is None:
        q = np.random.randn(d, k + 1)
        q, _ = np.linalg.qr(q)
    errors = np.zeros(t)

    for j in range(t):
        sine = np.linalg.norm(np.matmul(np.identity(d) - np.matmul(vk.transpose(), vk), q[:, 0:k]), ord=2)
        # sine = np.dot(q[:, k], vk)
        errors[j] = sine

        z = np.zeros((d, k + 1))
        for i in range(m):
            qi, zi, _ = local_power(data[i], k, p, q=q, vk=vk)
            if i == 0:
                z0 = zi
            else:
                D = procrustes(zi, z0)
                qi = np.matmul(qi, D)
            z += qi
        q, _ = np.linalg.qr(z)

        if is_decay:
            if exp:
                p = max(1, p//2)
            else:
                p = max(1, p - 1)
    return q, errors


def DR_SVD(data, k, iter=0, vk=None):
   n, d = data.shape
   r = k + ((d-k)//4)

   Omega = np.random.normal(size=(d, r))
   Y = np.matmul(data, Omega)
   for i in range(iter):
       Y = data @ (data.T @ Y)

   Q, _ = np.linalg.qr(Y)

   B = np.matmul(Q.T, data)
   _, _, V = np.linalg.svd(B)
   V = V.T
   error = np.linalg.norm(np.matmul(np.identity(d) - np.matmul(vk.transpose(), vk), V[:, 0:k + 1]))

   return error


def UDA(data, k, vk=None):
    """
    Args:
        data: a list of m matrices, each of which is s-by-d
        k: target dimension
        vk: d-by-k target eigenvector matrix
    """

    m = len(data)
    d = data[0].shape[1]

    z = np.zeros((d,d))
    for i in range(m):
        _, D, V = np.linalg.svd(np.matmul(data[i].T, data[i]))
        V = V.T
        z += np.matmul(V[:, 0:k+1], V[:, 0:k+1].T)
        V = V.T

    _, _, V = np.linalg.svd(z)
    V = V.T
    error = np.linalg.norm(np.matmul(np.identity(d) - np.matmul(vk.transpose(), vk), V[:, 0:k + 1]))

    return error


def WDA(data, k, vk=None):
    """
    Args:
        data: a list of m matrices, each of which is s-by-d
        k: target dimension
        vk: d-by-k target eigenvector matrix
    """

    m = len(data)
    d = data[0].shape[1]

    z = np.zeros((d, d))
    for i in range(m):
        _, D, V = np.linalg.svd(np.matmul(data[i].T, data[i]))
        D_ = np.diag(D[0:k+1])
        V = V.T
        z += np.matmul(V[:, 0:k+1], np.matmul(D_, V[:, 0:k+1].T))

    _, _, V = np.linalg.svd(z)
    V = V.T
    error = np.linalg.norm(np.matmul(np.identity(d) - np.matmul(vk.transpose(), vk), V[:, 0:k + 1]))
    return error
