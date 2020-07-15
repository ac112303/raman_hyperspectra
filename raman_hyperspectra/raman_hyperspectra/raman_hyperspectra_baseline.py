"""
Baseline extraction of a spectra or an hyperspectra using:
    - a morphological filter white tophat or;
    - an improved asymmetric least squares (ials)or;
    - an asymmetric least square (als)or;
    - asymmetrically reweighted penalized least squares smoothing (arPLS) or;
    - a linear baseline or;
    - the rubberband method or;
    - Modpoly : for full information.see BaselineRemoval package or  https://doi.org/10.1366/000370203322554518 ; 
    - Imodpoly : for full information.see BaselineRemoval package or https://doi.org/10.1366/000370207782597003

Internal functions:
  supress_baseline : linear baseline removal implemementation
  topHat : tophat morphological tophat implementation
  baseline_als : asymmetric least square implemementation
  baseline_ials : improved asymmetric least square implemementation
  baseline_arPLS : asymmetrically reweighted penalized least squares smoothing (arPLS)
  baseline_drPLS :  doubly reweighted penalized least squares
  rubberband : https://dsp.stackexchange.com/questions/2725/how-to-perform-a-rubberband-correction-on-spectroscopic-data

3rd party dependencies : numpy, scipy, tqdm

"""

__all__ = ["flatten_hyperspectra", "flatten_spectra"]


def flatten_hyperspectra(da, method):

    """
    Baseline supression of all the spectra of an hyperspectra.
    
    Arguments :
        da (xarray): hyperspectrum to flatten
        method (tuple): (method,*param)
        
    Returns:
        da (xarray):  flattened hyperspectrum
    """

    # 3rd party imports
    import numpy as np
    from tqdm import trange
    from BaselineRemoval import BaselineRemoval

    # Local application import
    from raman_hyperspectra.raman_hyperspectra_read_files import construct_xarray

    method_type, *param = method

    X = da.data.reshape((da.shape[0] * da.shape[1], da.shape[2]))

    zth = []
    if method_type == "linear":
        for i in trange(da.shape[0] * da.shape[1], desc="linear flattening"):
            z, _ = supress_baseline(da.lbd.values, X[i, :], *param)
            zth.append(z)

    elif method_type == "top hat":
        top_hat_bandwith = param[0]
        factor = top_hat_bandwith / da.lbd.attrs["spectral_resolution"] / len(da.lbd)
        for i in trange(da.shape[0] * da.shape[1], desc="tophat flattening"):
            zth.append(topHat(X[i, :], factor=factor))

    elif method_type == "ials":
        for i in trange(da.shape[0] * da.shape[1], desc="ials flattening"):
            z_base = baseline_ials(da.lbd.values, X[i, :], *param)
            zth.append(X[i, :] - z_base)

    elif method_type == "als":
        for i in trange(da.shape[0] * da.shape[1], desc="als flattening"):
            z_base = baseline_als(X[i, :], *param)
            zth.append(X[i, :] - z_base)

    elif method_type == "arPLS":
        for i in trange(da.shape[0] * da.shape[1], desc="arPLS flattening"):
            z_base = baseline_arPLS(X[i, :], *param)
            zth.append(X[i, :] - z_base)

    elif method_type == "drPLS":
        for i in trange(da.shape[0] * da.shape[1], desc="drPLS flattening"):
            z_base = baseline_drPLS(X[i, :], *param)
            zth.append(X[i, :] - z_base)

    elif method_type == "rubberband":
        for i in trange(da.shape[0] * da.shape[1], desc="rubberband flattening"):
            z_base = rubberband(da.lbd.values, X[i, :])
            zth.append(X[i, :] - z_base)

    elif method_type == "Modpoly":
        for i in trange(da.shape[0] * da.shape[1], desc="Modpoly flattening"):
            baseObj = BaselineRemoval(X[i, :], *param)
            Modpoly_output = baseObj.ModPoly()
            zth.append(X[i, :] - z_base)

    elif method_type == "Imodpoly":
        for i in trange(da.shape[0] * da.shape[1], desc="Imodpoly flattening"):
            baseObj = BaselineRemoval(X[i, :], *param)
            Imodpoly_output = baseObj.IModPoly()
            zth.append(X[i, :] - z_base)

    else:
        raise Exception(f"unknown method:{method_type}")


    try:
        tool = da.attrs["tool"]

    except:
        tool = "unknown"

    try:
        hyperspectra_name = da.attrs["hyperspectra_name"]

    except:
        tool = "unknown"

    da = construct_xarray(
        np.array(zth).reshape((da.shape)),
        range(da.shape[0]),
        range(da.shape[1]),
        da["lbd"].data,
        tool=tool,
        hyperspectra_name=hyperspectra_name,
    )
    da.attrs["flattening method"] = method_type

    return da


def flatten_spectra(da, method):

    """
    flattening of a spectrum
    
    Arguments :
        da (xarray): spectrum xarray
        method (tuple): tuple (method,*param)
        
    Returns:
        z (np array): flattened spectrum
        z_base (np array): baseline
    """

    # 3rd party dependecies
    from BaselineRemoval import BaselineRemoval

    method_type, *param = method

    if method_type == "linear":
        z, z_base = supress_baseline(da.lbd.values, da.values, *param)

    elif method_type == "top hat":
        top_hat_bandwith = param[0]
        factor = top_hat_bandwith / da.lbd.attrs["spectral_resolution"] / len(da.lbd)
        z = topHat(da, factor=factor)
        z_base = da.values - z

    elif method_type == "ials":
        z_base = baseline_ials(da.lbd.values, da.values, *param)
        z = da.values - z_base

    elif method_type == "als":
        z_base = baseline_als(da.values, *param)
        z = da.values - z_base

    elif method_type == "arPLS":
        z_base = baseline_arPLS(da.values, *param)
        z = da.values - z_base

    elif method_type == "drPLS":
        z_base = baseline_drPLS(da.values, *param)
        z = da.values - z_base

    elif method_type == "rubberband":
        z_base = rubberband(da.lbd.values, da.values)
        z = da.values - z_base

    elif method_type == "Modpoly":
        baseObj = BaselineRemoval(da.values, *param)
        z = baseObj.ModPoly()
        z_base = da.values - z

    elif method_type == "Imodpoly":
        baseObj = BaselineRemoval(da.values, *param)
        z = baseObj.IModPoly()
        z_base = da.values - z

    else:
        raise Exception(f"unknown method:{method_type}")

    return z, z_base


def topHat(data, factor=0.4):

    """
    Supress the baseline using a top hat filter 
    source: https://doi.org/10.1366/000370210791414281
    
    Arguments :
     data (nparray): Raman spectrum intensity values
     factor (real): relative length of the structure element versus the length of the spectra
     
    Returns:
     data_flatten (np array):  flattened spectrum

     """

    # 3rd party imports
    from scipy import ndimage
    import numpy as np

    str_el = np.repeat([1], int(round(data.size * factor)))
    data_flatten = ndimage.white_tophat(data, structure=str_el)

    return data_flatten


def rubberband(lbd, z):
    """
    method: https://dsp.stackexchange.com/questions/2725/how-to-perform-a-rubberband-correction-on-spectroscopic-data
    
    Arguments:
        lbd (nparray): wavelengths
        z (nparray): spectrum
        
    Returns:
        baseline (nparray): baseline
    """

    # 3rd party imports
    from scipy.spatial import ConvexHull
    import numpy as np

    v = ConvexHull([[X[0], X[1]] for X in zip(lbd, z)]).vertices
    v = np.roll(
        v, -v.argmin()
    )  # Rotate convex hull vertices until they start from the lowest one
    v = v[: v.argmax()]  # Leave only the ascending part

    baseline = np.interp(
        lbd, lbd[v], z[v]
    )  # Create baseline using linear interpolation between vertices

    return baseline


def baseline_drPLS(y, lam=1.0e6, eta=0.5, ratio=0.001, iter_max=100):

    """
    method : http://ao.osa.org/abstract.cfm?URI=ao-58-14-3913
    
    Arguments:
        y (nparray): spectra
        lam (float): ponderation of the first derivative penalty (default = 1.e6)
        ratio (float):
        eta (float):
        iter_max (int): maximum number of iterations
    
    Returns:
        baseline (nparray): baseline
    """

    # 3rd party imports
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    L = len(y)

    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), format="csr")
    D = D.dot(D.transpose())
    D_1 = sparse.diags([-1, 1], [0, -1], shape=(L, L - 1), format="csr")
    D_1 = D_1.dot(D_1.transpose())

    w_0 = np.ones(L)
    I_n = sparse.diags(w_0, format="csr")

    w = w_0
    W = sparse.diags(w, format="csr")
    Z = w_0

    for jj in range(iter_max):
        W.setdiag(w)
        Z_prev = Z
        Z = sparse.linalg.spsolve(
            W + D_1 + lam * (I_n - eta * W) * D, W * y, permc_spec="NATURAL"
        )

        if np.linalg.norm(Z - Z_prev) > ratio:
            d = y - Z
            d_negative = d[d < 0]
            sigma_negative = np.std(d_negative)
            mean_negative = np.mean(d_negative)
            w = 0.5 * (
                1
                - np.exp(jj)
                * (d - (-mean_negative + 2 * sigma_negative))
                / sigma_negative
                / (
                    1
                    + np.abs(
                        np.exp(jj)
                        * (d - (-mean_negative + 2 * sigma_negative))
                        / sigma_negative
                    )
                )
            )

        else:
            break

    baseline = Z

    return baseline


def baseline_arPLS(y, lam=1.0e5, ratio=1.0e-3, iter_max=40):

    """
    Baseline correction using asymmetrically reweighted penalized least squares smoothing (arPLS)
    from  http://dx.doi.org/10.1039/C4AN01061B
    
    Arguments:
        y    (ndarray): intensity array
        lam  (float)  : ponderation of the first derivative penalty (default = 1.e5)
        ratio (float) : exit criteria (default = 1.e-3)
        iter_max (int) : maximum number of ieration (default= 40)
        
    Returns:
        baseline (nparray): baseline
    """

    # 3rd party imports
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    iter_ = 0

    while True:
        W = sparse.spdiags(w, 0, L, L)
        Z = W + D
        baseline = spsolve(Z, W * y)
        d = y - baseline
        d_neg = d[d < 0]
        m = np.mean(d_neg)
        s = np.std(d_neg)

        sigmud_arg = 2 * (d - (2 * s - m)) / s  # avoid numerical overflow in exp
        wt = np.array([1 / (1 + np.exp(arg)) if arg < 200 else 0 for arg in sigmud_arg])

        if (np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio) or (iter_ > iter_max):
            break
        w = wt
        iter_ += 1

    return baseline


def baseline_ials(lbd, y, lam, lam1, p):

    """
    Baseline correction using improved asymmetric least square IAsLS
    from  https://doi.org/10.1366/000370210791414281
  
    Args:
        lbd (np array  : vawelength array
        y (np array)   : intensity array
        lam (real) : ponderation of the second derivative penalty (typical 10000)
        lam1 (real) : ponderation of the first derivative penalty (typical 0.01)
        p    : p asymmetric coefficient 0<p<1 (typical 0.001)
        
    Returns
        baseline (np array)    : baseline
    """

    # 3rd party imports
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    y = np.array(y)
    L = len(y)

    D1 = sparse.diags([1, -1], [0, -1], shape=(L, L - 1))
    D1 = lam1 * D1.dot(D1.transpose())

    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())

    zp = np.polyfit(lbd, y, 2)
    P = np.poly1d(zp)
    z = P(lbd)

    for _ in range(2):

        zp = np.polyfit(lbd, y, 2)
        P = np.poly1d(zp)
        z = P(lbd)
        w = p * (y > z) + (1 - p) * (y < z)

        for _ in range(4):
            W = sparse.spdiags(w, 0, L, L)
            W = W.dot(W.transpose())
            Z = W + D1 + D
            z = spsolve(Z, (W + D1) * y)
            w = p * (y > z) + (1 - p) * (y <= z)
        y = z
    baseline = z

    return baseline


def baseline_als(y, lam, p, niter):

    """
    Baseline correction using asymmetric least square (AsLS)
    from  https://doi.org/10.1366/000370210791414281
    
    Arguments:
        y    : intensity array
        lam  : ponderation of the second derivative penalty (typical 1.e5)
        p    : p asymmetric coefficient 0<p<1 
        niter: maximum number of iterations (typical 40)
        
    Returns:
        baseline (nparray): baseline
    """

    # 3rd party imports
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(
        D.transpose()
    )  # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values
        Z = W + D
        baseline = spsolve(Z, w * y)
        w = p * (y > baseline) + (1 - p) * (y < baseline)

    return baseline


def supress_baseline(lbd, z, lbd_base_search, npts_mean):

    """
    baseline correction with a linear function z0+(z1-z0)*(lbd-lbd0)/(lbd1-lbd0)
    
    Arguments:
        lbd (nparray):  vawelengths values
        z (nparray):  spectrum intensity values
        lbd_base_search : lbd0 = argmin(z[lbd[0] : lbd_base_search])
                          lbd0 = lbd[0] if lbd_base_search is None 
        npts_mean (int): number or points to evaluate z0 and z1 as the mean of z
        
    Returns:
        z (nparray): flatten spectrum
        z_base (nparray) :  baseline
    """

    # Standard Library imports
    import bisect

    # 3rd party imports
    import numpy as np

    if lbd_base_search is None:
        lbd0 = lbd[0]
        z0 = np.mean(z[0:npts_mean])

    else:  # lbd0 is the the vawelength of the minimum intensity on the interval [lbd[0], lbd_base_search]
        idx_sup_search_min = bisect.bisect_right(lbd, lbd_base_search)
        jk_min = np.argmin(z[0:idx_sup_search_min])
        lbd0 = lbd[jk_min]
        z0 = np.mean(z[jk_min : jk_min + npts_mean])

    lbd1 = lbd[-1]
    z1 = np.mean(z[-npts_mean:])

    # correction of the linebase to satisfie intensity >= 0
    z_base = z0 + (z1 - z0) * (lbd - lbd0) / (lbd1 - lbd0)
    z_base = z_base + min( z - z_base)
 
    z = z - z_base

    return z, z_base
