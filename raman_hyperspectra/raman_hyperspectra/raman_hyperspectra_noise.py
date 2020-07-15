"""
Std noise evaluation of a spectrum/hyperspectrum. The noise is obtained by substracting the signal 
with the low pass filtered signal. The low pass can be:
    - a Savitky_Golay moving smoothig or;
    - a gauss filter

3rd party dependencies : 
numpy, xarray, matplotlib, scipy, raman_hyperspectra.raman_hyperspectra_read_files
"""

__all__ = ["noise_plot", "spectra_noise_estimation", "spectra_noise_estimation_0D"]


def noise_plot(
    da_denoise,
    da_raw,
    da_noise,
    bins=50,
    nbr_std=5,
    style="ggplot",
    save=True,
    robust=True,
) -> None:

    """
    plot the noise histogram. the noise is computed as da_denoise - da_raw  where da_raw is
    the unprocessed hyperspecta and da_denoise is the denoised hyperspectra
    
    Arguments:
        da_denoise (xarray): denoised hyperspectra
        da_raw (xarra): raw hyperspectra
        bins (integer):  number of bins used to draw the noise histogram (default=50)
        nbr_std (integer): fixes the upper and lower limits of the histogram to +/- nbr_std * the noise standard deviation (std)
        style(str) : plot style
        save (bool): if save is True the histogram plot is save as noise_histogram_hh_mm_ss.png
    Returns:
    
    """

    import numpy as np
    import xarray as xr
    import sys
    import matplotlib.pyplot as plt
    import time

    def plot_histo() -> None:
        noise = (
            da_denoise.values - da_raw.values[:, :, 0 : da_denoise.shape[2]]
        ).flatten()
        noise = noise - np.mean(noise)
        std = np.std(noise)
        noise = noise[noise < nbr_std * std]
        noise = noise[noise > -nbr_std * std]

        fig = plt.figure(figsize=(10, 5))
        plt.style.use(style)

        plt.subplot(1, 2, 1)
        plt.hist(noise, bins=bins, edgecolor="black")
        std = round(np.std(noise), 2)
        plt.title(f"standard deviation = {std}")

        plt.subplot(1, 2, 2)
        np.max(da_denoise, axis=2).plot(robust=robust)
        max_hight_peak = np.max(da_denoise.values)
        plt.title(f"maximum = {round(max_hight_peak,1)}")
        plt.gca().set_aspect("equal", adjustable="box")
        if save:
            fig = plt.gcf()
            fig.savefig(
                sys._getframe().f_code.co_name
                + " "
                + time.strftime("%H_%M_%S")
                + ".png"
            )
        plt.show()

    def plot_noises():
        da_std = np.std(da_denoise - da_raw, axis=2)
        da = xr.DataArray(
            np.array([da_std.values]),
            dims=["Filter", "y", "x"],
            name="Intensity",
            attrs={"units": "u.a."},
            coords={
                "Filter": xr.DataArray(
                    ["NRWRT"], name="Filter", dims=["Filter"], attrs={"units": ""}
                ),
                "y": xr.DataArray(
                    da_std.x.values, name="y", dims=["y"], attrs={"units": "u.a."}
                ),
                "x": xr.DataArray(
                    da_std.y.values, name="x", dims=["x"], attrs={"units": "u.a."}
                ),
            },
        )

        da = xr.concat([da_noise, da], "Filter")
        t = da.sel(Filter=da.Filter.values)
        t.plot(x="x", y="y", col="Filter", col_wrap=3, robust=robust)
        if save:
            fig = plt.gcf()
            fig.savefig(
                sys._getframe().f_code.co_name
                + " "
                + time.strftime("%H_%M_%S")
                + ".png"
            )
        plt.show()

        return da

    plot_histo()

    da = plot_noises()

    return da


def spectra_noise_estimation(
    da, savgol_filter_, gaussian_filter_, lbd_inf_var=100, lbd_sup_var=480
):

    """
    std estimation of an hyperspectra. 
    The noise of each spectra is obtained using:
         noise[lbd_inf_var:lbd_sup_var] = spectra[lbd_inf_var:lbd_sup_var] - LowPass(soectra[lbd_inf_var:lbd_sup_var])
    The LowPass(signal) is obtained (i) using a savgol_filter  or a Gaussian filter. 
    The std is estimated using np.std(noise) and the result are stored in an xarray
    
    Arguments:
        da (xarray): hyperspectra 
        savgol_filter_ (tuple): (("Savitzky-Golay",length,order)) second method used to compute the std of a spectrum
        gaussian_filter_ (tuple): (("Gauss",sigma)) first method used to compute the std of a spectrum
        lbd_inf_var (real): the std is evaluated on the bandwidth [lbd_inf_var, lbd_sup_var]
        lbd_inf_var (real): the std is evaluated on the bandwidth [lbd_inf_var, lbd_sup_var]
        
    Returns:
        da_noise (xarray): xarray std for both methods savgol_filter_ and gaussian_filter_

    """

    # 3rd party dependencies
    import xarray as xr
    from scipy.signal import savgol_filter
    from scipy.ndimage.filters import gaussian_filter1d
    from raman_hyperspectra.raman_hyperspectra_read_files import construct_xarray
    import numpy as np

    db = da.sel(lbd=slice(lbd_inf_var, lbd_sup_var))

    # build a std xarray using the savgol_filter method
    _, *param = savgol_filter_
    db_sg = construct_xarray(
        savgol_filter(db.values, param[0], param[1], axis=2, mode="constant"),
        db.x.values,
        db.y.values,
        db.lbd.values,
    )

    std_savgol_filter = np.std(db.values - db_sg.values, axis=2)

    # build a std xarray using the gaussian_filter method
    _, *param = gaussian_filter_
    db_g = construct_xarray(
        gaussian_filter1d(db.values, param[0], axis=2, mode="constant"),
        db.x.values,
        db.y.values,
        db.lbd.values,
    )

    std_gaussian_filter = np.std(db.values - db_g.values, axis=2)

    # build an xarray merging std_savgol_filter and std_gaussian_filter
    x = da.x.values
    y = da.y.values

    da_noise = xr.DataArray(
        np.array([std_savgol_filter, std_gaussian_filter]),
        dims=["Filter", "y", "x"],
        name="Intensity",
        attrs={"units": "u.a."},
        coords={
            "Filter": xr.DataArray(
                ["Savitzky-Golay", "Gauss"],
                name="Filter",
                dims=["Filter"],
                attrs={"units": ""},
            ),
            "y": xr.DataArray(
                y, name="y", dims=["y"], attrs={"units": da.y.attrs["units"]}
            ),
            "x": xr.DataArray(
                x, name="x", dims=["x"], attrs={"units": da.x.attrs["units"]}
            ),
        },
    )

    return da_noise


def spectra_noise_estimation_0D(
    da, method1, method2, lbd_dep=None, lbd_end=None, n_split=1
):

    """
    std estimation of a spectra. 
    The noise of the spectra is obtained using:
         noise[lbd_inf_var:lbd_sup_var] = spectra[lbd_inf_var:lbd_sup_var] - LowPass(soectra[lbd_inf_var:lbd_sup_var])
    where the LowPass(signal) is obtained (i) using a savgol_filter  or a Gaussian filter. 
    The std is estimated using the clever variance algorithm:'Outlier detection in large data sets' G. Buzzi-Ferraris and
    F. Manenti http://dx.doi.org/10.1016/j.compchemeng.2010.11.004
    
    Arguments:
        da : xarray containing the spectra
        method1 (tuple): tuple(method,*param) first method used to compute the std of a spectrum
        method2 (tuple): tuple(method,*param) second method used to compute the std of a spectrum
        lbd_dep (float): the std is evaluated on the bandwidth [lbd_dep, lbd_end]
        lbd_end (float): the std is evaluated on the bandwidth [lbd_dep, lbd_end]
        n_split (int): number of slices of equal length on which the noise is computed (default =1)
        
    Returns:
        std_svg_mean (float): mean malue of the std_svg
        std_g_mean (float): mean malue of the std_g
        std_svg (list): list std obtained with method1 on each slice with savgol_filter filter
        std_g (list): std obtained with method2 on each slice with gaussian filter
        noise_svg (list of nparray): array of the noise on each slice with savgol_filter filter 
        noise_g (list of nparray): array of the noise on each slice  with gaussian filter
        lbd_noise_split(list of nparray):list of the wavelength on each slice
        z_raw_split (list of nparray): list of the raw spectra on each slice
         
    """

    # Internal dependencies
    from raman_hyperspectra.raman_hyperspectra_denoise import smooth_spectrum

    # 3rd party dependencies
    import numpy as np

    lbd = da.lbd.values

    if not lbd_dep:
        lbd_dep = lbd[0]
    if not lbd_end:
        lbd_end = lbd[-1]

    assert lbd_end > lbd_dep, "lbd_dep must be lower than lbd_end"

    if method1[0] == "Gauss":  # force method1 to be Savitzky-Golay
        (method1, method2) = (method2, method1)

    db = da.sel(lbd=slice(lbd_dep, lbd_end))

    # drop elements so that the length of z_raw and lbd_noise
    # are a mulipule of n_split
    long = db.shape[0] - db.shape[0] % n_split
    lbd_noise = db.lbd.values[:long]
    z_raw = db.values[:long]

    # compute the clever std on each of the n_split slices
    std_svg, std_g, noise_svg, noise_g, lbd_noise_split, z_noise_split = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for lbd_split, z_raw_split in zip(
        np.split(lbd_noise, n_split), np.split(z_raw, n_split)
    ):

        noise_svg_split = z_raw_split - smooth_spectrum(z_raw_split, method1)
        std_svg_split, _ = clever_variance(
            lbd_split, noise_svg_split, number_max_outliers=None
        )
        std_svg.append(std_svg_split)
        noise_svg.append(noise_svg_split)

        noise_g_split = z_raw_split - smooth_spectrum(z_raw_split, method2)
        std_g_split, _ = clever_variance(
            lbd_split, noise_g_split, number_max_outliers=None
        )
        std_g.append(std_g_split)
        noise_g.append(noise_g_split)

        lbd_noise_split.append(lbd_split)
        z_noise_split.append(lbd_split)
        
        std_svg_mean = np.mean(std_svg)
        std_g_mean = np.mean(std_g)

    return std_svg_mean, std_g_mean,std_svg, std_g, noise_svg, noise_g, lbd_noise_split, z_raw_split


def clever_variance(lbd, x, number_max_outliers=None):

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

            else:
                x_tup.insert(idx, xx)
                idx += 1
        except:
            pass

    if outliers:
        _, y = zip(*sorted(outliers, key=lambda y: y[1], reverse=False))
        outliers = list(y)
    else:
        outliers = []

    return np.sqrt(cv0), outliers
