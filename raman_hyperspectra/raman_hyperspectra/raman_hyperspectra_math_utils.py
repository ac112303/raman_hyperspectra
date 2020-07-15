# binning
# find_outliers

__all__ = []


def binning(X, binx, biny):
    """Binning of 2D array X"""
    if binx != 1 and biny != 1:
        if np.remainder(X.shape[1], binx) != 0 and np.remainder(X.shape[2], biny) != 0:
            raise Exception("Invalid binning factor")
        X = np.array(
            [
                np.array(x)
                for x in [
                    X[I, :, :]
                    .reshape(X.shape[1] // binx, binx, X.shape[2] // biny, biny)
                    .mean(-1)
                    .mean(1)
                    for I in range(X.shape[0])
                ]
            ]
        )
    return X


def savitsky_golay_mat(N, M):
    import numpy as np

    z = np.linspace(-(N // 2), N // 2, N)
    G = np.ones(N)
    if M > 0:
        for i in range(1, M + 1):
            G = np.vstack((G, z ** i))
        C = np.matmul(np.linalg.inv(np.matmul(G, G.T)), G)
    else:
        C = 1 / N * z

    return C
