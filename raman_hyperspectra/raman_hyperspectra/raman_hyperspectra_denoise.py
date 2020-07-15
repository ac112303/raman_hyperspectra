"""
Denoising of an spectrum/hyperspectrum using three methods:
  - Savitzky-Golay moving smoothing( https://doi.org/10.1021/ac60214a047):
          (method= 'Savitzky-Golay',
                   interval length,
                   polynome order)
  - wavelet denoising (https://link.springer.com/book/10.1007/978-1-4613-0145-5):
           (method ='NRWRT',
            spectrum std ,
            number of decimation, wavelet type http://wavelets.pybytes.com/,
            threshold choice,
            ponderation coefficient of the universal thresholding' (ThU))

Intrnal functions:
  denoise_hyperspectra
  denoise_wavelet


3rd party dependencies : pywt, numpy, scipy, tqdm
"""


__all__ = ["denoise_hyperspectra", "smooth_spectrum"]


def denoise_hyperspectra(da, method):

    """
    Hyperspectrum denoising  using NRWRT or Savitzky-Golay or moving_average

    Arguments:
        da (xarray): hyperspectra
        method (tuple): (method, *param)
                if method == NRWRT
                             param[0] = da_std : array containing the std noise of the spectra
                             param[1] = decNum = nombre de décimations pour le filtrage MRA
                             param[2] = wname : wavelett type,
                             param[3] = mode = choix de seuillage (soft/hard)
                             param[4] = ratio_sigma : the ratio of the std use to the threshold denoising
                if method == Savitzky-Golay
                             param[0] = interval length (must be odd)
                             param[1] = polynome order
                if method == moving_average
                             param[0] = filter length

    Returns:
        da (xarray): denoised hyperspectrum

    """

    # 3rd party imports
    import numpy as np
    from tqdm import trange
    from scipy.signal import savgol_filter

    # Local application imports
    from raman_hyperspectra.raman_hyperspectra_read_files import construct_xarray

    method_type, *param = method

    try:
        tool = da.attrs["tool"]

    except:
        tool = "unknown"

    try:
        hyperspectra_name = da.attrs["hyperspectra_name"]

    except:
        hyperspectra_name = "unknown"

    # flatten xarray in spataial coordinates to save one loop
    X = da.data.reshape((da.shape[0] * da.shape[1], da.shape[2]))

    if method_type == "NRWRT":

        d_std = param[0]
        std = d_std.reshape((d_std.shape[0] * d_std.shape[1]))

        Y = []
        for i in trange(
            da.shape[0] * da.shape[1], desc=f"denoising with {method_type}"
        ):

            zd = denoise_wavelet(
                X[i, :],
                std[i],
                decNum=param[1],
                wname=param[2],
                mode=param[3],
                ratio_sigma=param[4],
            )

            if len(zd) == da.shape[2]:  # complete the zd array to match the lbd size
                Y.append(zd)

            else:
                Y.append(zd[0:-1])

        da = construct_xarray(
            np.array(Y).reshape((da.shape[0], da.shape[1], -1)),
            range(da.shape[0]),
            range(da.shape[1]),
            da["lbd"].data[0 : np.shape(Y)[1]],
            tool=tool,
            hyperspectra_name=hyperspectra_name,
        )

        da.attrs[
            "denoising method"
        ] = f"NRWRT decNum = {param[1]} , wname = {param[2]}, \
                                        mode = {param[3]}, ratio_sigma = {param[4]}"

    elif method_type == "Savitzky-Golay":

        da = construct_xarray(
            savgol_filter(da.values, *param, axis=2, mode="constant"),
            da.x.values,
            da.y.values,
            da.lbd.values,
            tool=tool,
            hyperspectra_name=hyperspectra_name,
        )
        da.attrs[
            "denoising method"
        ] = f"Savitzky-Golay len = {param[0]} , order = {param[1]}"

    elif method_type == "Moving Average":

        Y = []
        for i in trange(
            da.shape[0] * da.shape[1], desc=f"denoising with {method_type}"
        ):

            length = 2 * (param[0] // 2) + 1  # change even to odd length
            zd = lfilter([1.0 / length] * length, 1, X[i, :])
            np.roll(zd, length // 2)  # cancell the delay

        da = construct_xarray(
            np.array(Y).reshape((da.shape[0], da.shape[1], -1)),
            range(da.shape[0]),
            range(da.shape[1]),
            da["lbd"].data[0 : np.shape(Y)[1]],
            tool=tool,
            hyperspectra_name=hyperspectra_name,
        )
        da.attrs["denoising method"] = f"moving average len = {param[0]}"

    return da


def smooth_spectrum(z, method, std=0):

    """
    spectrum smoothing

    Arguments:
        z (nparray): spectrum intensities
        method (tuple): (method, *param)
                if method == NRWRT
                             param[0] = decNum = nombre de décimations pour le filtrage MRA
                             param[1] = wname : type d'ondelette,
                             param[2] = mode = choix de seuillage (soft/hard)
                             param[3] = ratio_sigma : the ratio of the std use to the threshold denoising
                if method == Savitzky-Golay
                             param[0] = interval length (must be odd)
                             param[1] = polynome order
                if method == moving_average
                             param[0] = filter length

        std (real): std noise estimation of the spectrum (used by NRWRT)

    Returns:
        z (np array): denoided spectra
    """

    # 3rd party imports
    from scipy.signal import lfilter
    from scipy.signal import savgol_filter
    from scipy.ndimage.filters import gaussian_filter1d
    import numpy as np

    method_type, *param = method

    if method_type == "Savitzky-Golay":
        z_smooth = savgol_filter(z, *param, mode="nearest")

    elif method_type == "Moving Average":
        length = 2 * (param[0] // 2) + 1  # change even to odd length
        z_smooth = lfilter([1.0 / length] * length, 1, z)
        z_smooth = np.roll(z_smooth, -(length // 2))  # cancell the delay

    elif method_type == "NRWT":
        z_smooth = denoise_wavelet(z, std, *param)

        if len(z_smooth) != len(z):
            z_smooth = z_smooth[0:-1]

    elif method_type == "Gauss":
        z_smooth = gaussian_filter1d(z, param[0])

    return z_smooth


def denoise_wavelet(z, std, decNum=4, wname="db4", mode="soft", ratio_sigma=0.9):

    """
    Wavelet denoising using noise reduction wavelet thresholdind

    Arguments:
        z (ndarray): signal to denoise
        std (float): std of the spectrum noise
        decNum (int): decimations number for the MRA filter
        wname (string): wavelett type
        mode (string): thresholding method (soft/hard/garrote)
        ratio_sigma (float): ponderation coefficient of std

    Returns:
        da (ndarray): denoised signal

    """

    # 3rd party imports
    import numpy as np
    import pywt

    sigma = ratio_sigma * std

    t = sigma * np.sqrt(2 * np.log(len(z)))  # universal threshold(ThU) computation

    coeffs = pywt.wavedec(z, wname, level=decNum)  # wavelett decomposition
    coefft = []

    for ii in range(
        decNum + 1
    ):  # wavelett coefficient thresholding using mode=soft/hard/garrote
        coefft.append(pywt.threshold(coeffs[ii], t, mode))

    zd = pywt.waverec(
        coefft, wname
    )  # spectra rebuilding using the trimmed coefficients

    return zd
