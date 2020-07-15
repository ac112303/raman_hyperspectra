"""
Implementation of the PCA, robust PCA, NMF on an hypespectrum.
PCA or robust PCA are used to remove the cosmic spikes from an hypespectra.

Internal funtions:
NMF_decomposition

3rd party dependencies :
numpy, raman_hyperspectra.raman_hyperspectra_read_files,sklearn, xarray, matplotlib, sporco

"""

__all__ = [
    "Raman_NMF",
    "Raman_PCA",
    "Raman_Robust_PCA",
    "cosmic_cleaning",
    "remove_spikes",
]


def cosmic_cleaning(da, Kmax=17):

    """
    cosmic_cleaning supress the cosmic spikes from the Raman hypespectra. The algorithm is
    based on PCA denoising

    Arguments:
     da (DataArray): hyperspectra
     Kmax (integer): number of components used to retroprojection (default =17)

    Returns:
     da (xarray): cleaned hyperspectra

    """

    # 3rd party imports
    import numpy as np

    # Local application imports
    from raman_hyperspectra.raman_hyperspectra_read_files import construct_xarray

    X = da.data.reshape((da.shape[0] * da.shape[1], da.shape[2]))
    Cor = np.dot(X.T, X)
    lbd, Eigen_vec = np.linalg.eig(Cor)
    w = sorted(list(zip(lbd, Eigen_vec.T)), key=lambda tup: tup[0], reverse=True)
    vp = np.array([x[0] for x in w])
    L = np.array([x[1] for x in w]).reshape(np.shape(Eigen_vec)).T

    F = np.real(np.matmul(X, L))
    Eigen_vec = np.real(Eigen_vec)
    Kmin = 0
    X = np.matmul(np.real(F[0:, Kmin:Kmax]), np.real(L.T[Kmin:Kmax, 0:]))

    try:
        tool = da.attrs["tool"]

    except:
        tool = "unknown"

    try:
        hyperspectra_name = da.attrs["hyperspectra_name"]

    except:
        hyperspectra_name = "unknown"

    da = construct_xarray(
        X.reshape(da.shape),
        range(da.shape[0]),
        range(da.shape[1]),
        da["lbd"].data,
        tool=tool,
        hyperspectra_name=hyperspectra_name,
    )

    return da


def remove_spikes(da):

    """
    Cosmic_cleaning supress the cosmic spikes from the Raman spectra. The algorithm is
    based on the 'Outlier detection in large data sets' by G. Buzzi-Ferraris and
    F. Manenti http://dx.doi.org/10.1016/j.compchemeng.2010.11.004

    Arguments:
     da (DataArray): hyperspectra

    Returns:
     db (xarray): cleaned hyperspectra

    """

    # internal import
    from .raman_hyperspectra_read_files import construct_xarray_0D

    # 3rd party imports
    import numpy as np

    def find_outliers(lbd, x, number_max_outliers=None):

        """
        Method : [1] 'Outlier detection in large data sets' G. Buzzi-Ferraris and
        F. Manenti http://dx.doi.org/10.1016/j.compchemeng.2010.11.004

        Arguments:
            lbd (nparray): wavelength
            x (nparray): spectrum intensity

        Retuns:
            outliers (list of tuples): [(outlier value, outlier index, ),...]
        """

        # 3rd party imports
        import numpy as np

        if number_max_outliers is None:
            number_max_outliers = len(lbd)

        cv = lambda y: np.dot(y, y) / (len(y) - 1)  # clever variance [1] (4)

        x_tup = sorted(zip(x, range(len(x))), key=lambda y: y[0], reverse=True)

        outliers = []

        idx = 0
        for i in range(number_max_outliers):
            try:
                xx = x_tup.pop(idx)
                y, _ = zip(*x_tup)
                cm0 = np.mean(y)
                cv0 = cv(y - cm0)

                if abs(cm0 - xx[0]) > 2.5 * np.sqrt(cv0):  # condition [1] (7)
                    outliers.append(xx)
                    # print(cm0,cv0)

                else:
                    x_tup.insert(idx, xx)
                    idx += 1
            except:
                pass

        _, y = zip(*sorted(outliers, key=lambda y: y[1], reverse=False))
        outliers = list(y)

        return outliers

    def sieve_outliers(outliers_idx, z, n_consecutif=10, delta=10):

        """
        Selection of the "true" outliers. We reject cluster of outliers composed of at least
        n_consecutif consecutive outliers. In other word if the width of the cluster is greater
        than n_consecutif the cluster is considered to be a sharp peak of the spectra. The remaining peaks
        are supressed and their values replaced by a linear interploation.
        
        Arguments:
            outliers (list of tuples): [(outlier value, outlier index, ),...]
            z (nparray): spectra
            n_consecutif (int): if the number of consecutive outliers index is greater than n_consecutif
                                the outliers are not rejected (default = 10)
            delta (int): exclion zone around an outlier (default = 10)
        Returns:
            zz (nparray) : spectra without spikes
        """

        # partition of the outliers_idx in classes two indice pertain to the same class if their are consecutive
        reject = []  # list of array of consecutive indice
        consecutif = []  # an array of consecutive index
        for idx in range(1, len(outliers_idx)):
            if (
                outliers_idx[idx] - outliers_idx[idx - 1]
            ) == 1:  # check if two consecutive are consecutive
                consecutif.append(idx - 1)
            else:
                consecutif.append(idx - 1)
                reject.append(consecutif)
                consecutif = []

        consecutif.append(idx)  # take care of the last consecutive outliers
        reject.append(consecutif)

        # rejection of classes for whicch #classe > n_consecutif
        reject = np.array(
            [y for x in reject for y in x if len(x) > n_consecutif]
        )  # don't reject the peaks
        y = np.delete(outliers_idx, reject)

        # (i) removal of the outliers from the spectra; (ii) replace their values by a linear interplollation
        idx_to_exclude = [range(i - delta, i + delta) for i in y]
        lbd_index = np.array(range(len(z)))
        lbd_index_sieved = np.delete(lbd_index, idx_to_exclude)
        z_sieved = np.delete(z, idx_to_exclude)
        zz = np.interp(lbd_index, lbd_index_sieved, z_sieved)

        return zz

    lbd = da.lbd.values
    z = da.values
    outliers = find_outliers(lbd, z)
    zz = sieve_outliers(outliers, z, n_consecutif=10)

    try:
        tool = da.attrs["tool"]
    except:
        tool = "unkown"

    db = construct_xarray_0D(zz, lbd, tool=tool)

    return db


def NMF_decomposition(da, k):

    """
    non négative matrix decomposition

    Arguments:
        da (DataArray): xarray containing the hyperspectrum
        k (integer): number of componants

    Returns:
        W (nparray): non negative matrix
        H (nparray): signature matrix
    """

    # 3rd party imports
    from sklearn.decomposition import NMF
    import numpy as np

    A = da.data
    shape = da.data.shape
    A = A.reshape((shape[0] * shape[1], -1))

    M = np.where(A < 0, 0, A)  # mise à zéros des valeurs négatives
    Mn = []
    for i in range(np.shape(A)[0]):  # normalisation des spectres à 1
        Mn.append(M[i,] / np.max(M[i,]))
    np.reshape(Mn, np.shape(A))
    Mn = np.array(Mn)

    model = NMF(n_components=k, init="random", random_state=0)  # l1_ratio=1.0)
    W = model.fit_transform(Mn.T)
    H = model.components_

    return W, H


def Raman_NMF(da, k_NMF=3):

    """
    NMF decomposition of an hyperstra and plot of th k_NMF first components

    Arguments:
        da (DataArray): hyperstrum
        k_NMF (integer):number of components of the NMF decomposition

    Returns:
        W (np array): matrix W of the NMF decomposition
        H (np array): matrix H of the NMF decomposition

    """

    # Standard Library imports
    import sys
    import time

    # 3rd party imports
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt

    col_wrap = 3

    W, H = NMF_decomposition(da, k_NMF)

    shape = da.shape
    N_row = shape[0]
    N_col = shape[1]
    C2 = np.reshape(100 * H, (shape[0], shape[1], -1))
    dv = xr.DataArray(
        C2,
        dims=["y", "x", "Num_component"],
        name="NMF Intensity",
        attrs={"units": "u.a."},
        coords={
            "y": xr.DataArray(
                da.x.values, name="y", dims=["y"], attrs={"units": da.x.attrs["units"]}
            ),
            "x": xr.DataArray(
                da.y.values, name="x", dims=["x"], attrs={"units": da.y.attrs["units"]}
            ),
            "Num_component": xr.DataArray(
                range(np.shape(C2)[2]),
                name="Num_component",
                dims=["Num_component"],
                attrs={"units": ""},
            ),
        },
    )
    
    l = dv.isel(Num_component=range(k_NMF))
    g_simple = l.plot(
        x="x", y="y", col="Num_component", robust=False, col_wrap=col_wrap
    )

    fig = plt.gcf()
    fig.savefig(
        sys._getframe().f_code.co_name
        + "Intensity "
        + time.strftime("%H_%M_%S")
        + ".png"
    )
    plt.show()

    dc = xr.DataArray(
        W,
        dims=["lbd", "Num_component"],
        coords={
            "lbd": xr.DataArray(
                da.lbd.values, name="lbd", dims=["lbd"], attrs={"units": "cm-1"}
            ),
            "Num_component": xr.DataArray(
                range(np.shape(W)[1]),
                name="Num_component",
                dims=["Num_component"],
                attrs={"units": ""},
            ),
        },
    )
    l = dc.isel(Num_component=range(k_NMF))
    g_simple = l.plot(x="lbd", col="Num_component", col_wrap=col_wrap)

    fig = plt.gcf()
    fig.savefig(
        sys._getframe().f_code.co_name
        + "Intensity "
        + time.strftime("%H_%M_%S")
        + ".png"
    )
    plt.show()

    return W, H


def Raman_Robust_PCA(da):

    """
    robust PCA based on sporco module

    Arguments:
        da (DataArray): yperspectrum

    Returns:
        dX (DataArray): denoised hyperspectrum
        dY (DataArray): resual spike and noise
    """

    # 3rd party imports
    from sporco.admm import rpca
    import xarray as xr

    # Local application imports
    from raman_hyperspectra.raman_hyperspectra_read_files import construct_xarray

    opt = rpca.RobustPCA.Options(
        {
            "Verbose": True,
            "gEvalY": False,
            "MaxMainIter": 200,
            "RelStopTol": 5e-4,
            "AutoRho": {"Enabled": True},
        }
    )
    A = da.data
    shape = da.data.shape
    N_row = shape[0]
    N_col = shape[1]
    A = A.reshape((N_row * N_col, -1))
    b = rpca.RobustPCA(A, None, opt)
    X, Y = b.solve()

    dX = construct_xarray(
        X.reshape((N_row, N_col, -1)), da.x.values, da.y.values, da.lbd.values
    )
    dY = construct_xarray(
        Y.reshape((N_row, N_col, -1)), da.x.values, da.y.values, da.lbd.values
    )

    return dX, dY


def Raman_PCA(da, k_PCA=4):

    """
    PCA analysis of an hyperspectrum

    Arguments:
     da (DataArray): hyperspectra
     Kmax (int): (optional) number of components to be analalysed

    Returns:
     Eigen_value (nparray):  sorted eigenvalues
     F (nparray): score matrix
     L (nparray): load matrix

    """

    # Standard Library imports
    import time
    import sys

    # 3rd party imports
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt

    col_wrap = 3

    X = da.data.reshape((da.shape[0] * da.shape[1], da.shape[2]))
    Cor = np.dot(X.T, X)
    Eigen_value, Eigen_vec = np.linalg.eig(Cor)
    w = sorted(
        list(zip(Eigen_value, Eigen_vec.T)), key=lambda tup: tup[0], reverse=True
    )
    vp = np.array([x[0] for x in w])
    L = np.array([x[1] for x in w]).reshape(np.shape(Eigen_vec)).T

    F = np.real(np.matmul(X, L))
    Eigen_vec = np.real(Eigen_vec)

    plt.close()
    fig, ax = plt.subplots()
    ax.semilogy(sorted(100 * Eigen_value / sum(Eigen_value), reverse=True), "ro")
    ax.set_xlabel("n° component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("Scree plot")

    fig = plt.gcf()
    fig.savefig(
        sys._getframe().f_code.co_name + "Scree_ " + time.strftime("%H_%M_%S") + ".png"
    )
    plt.show()

    shape = da.shape
    N_row = shape[0]
    N_col = shape[1]
    C2 = np.reshape(np.real(F), (N_row, N_col, -1))
    dv = xr.DataArray(
        C2,
        dims=["y", "x", "Num_component"],
        name="PCA Intensity",
        attrs={"units": "u.a."},
        coords={
            "y": xr.DataArray(
                da.x.values, name="y", dims=["y"], attrs={"units": da.x.attrs["units"]}
            ),
            "x": xr.DataArray(
                da.y.values, name="x", dims=["x"], attrs={"units": da.y.attrs["units"]}
            ),
            "Num_component": xr.DataArray(
                range(np.shape(C2)[2]),
                name="Num_component",
                dims=["Num_component"],
                attrs={"units": ""},
            ),
        },
    )
    l = dv.isel(Num_component=range(k_PCA))
    l.plot(x="x", y="y", col="Num_component", robust=True, col_wrap=col_wrap)

    fig = plt.gcf()
    fig.savefig(
        sys._getframe().f_code.co_name
        + "Intensity "
        + time.strftime("%H_%M_%S")
        + ".png"
    )
    plt.show()

    dc = xr.DataArray(
        np.real(L),
        dims=["lbd", "Num_component"],
        coords={
            "lbd": xr.DataArray(
                da.lbd.values, name="lbd", dims=["lbd"], attrs={"units": "cm-1"}
            ),
            "Num_component": xr.DataArray(
                range(np.shape(L)[0]),
                name="Num_component",
                dims=["Num_component"],
                attrs={"units": ""},
            ),
        },
    )
    l = dc.isel(Num_component=range(k_PCA))
    l.plot(x="lbd", col="Num_component", col_wrap=col_wrap)

    fig = plt.gcf()
    fig.savefig(
        sys._getframe().f_code.co_name
        + "Intensity "
        + time.strftime("%H_%M_%S")
        + ".png"
    )
    plt.show()

    return Eigen_value, F, L
