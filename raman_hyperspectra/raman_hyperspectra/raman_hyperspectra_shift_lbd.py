"""
Correction for the wavelength shift of a spectrum using 
    - the location of the Rayleigh peak or;
    - the location of a well defined Raman peak.
    
    
Internal function:
    R2_plot
    fix_lbd0_py

3rd party dependencies : 
matplotlib, numpy, xarray, seaborn
"""

__all__ = ["set_rayleigh_peak_to_0", "set_peak_to_value", "update_xarray"]


def set_rayleigh_peak_to_0(
    db,
    file_init,
    sheet=None,
    tol_R2=0.99,
    lbd_deb_fit=-40,
    lbd_end_fit=40,
    lbd_end_flatten=70,
    bins=50,
    style="seaborn",
    save=True,
    robust=True,
):

    """
    Wavelength shift correction using the location of the Rayleigh peak
    
    Arguments:
        db (xarray): xarray containing the Rayleigh peak
        file_init (string): full path of the file containning the model
        sheet (string): name of the sheet (default = None)
        tol_R2 (real) : only the fit with R2> tol_R2 are used to compute the mean Rayleigh peak position (default =0.99)
        lbd_dep_fit (real): fitting lower wavelength in cm-1 (default = -40)
        lbd_end_fit (real): fitting upper wavelength in cm-1 (default = 40)
        lbd_end_flatten (real): upper wavelenght for the baseline removal in cm-1 (default = 70)
        bins (integer): number of bins for the histogramm
        style (string): plot style (default = 'seaborn')
        save (bool): if True saves the plot (default = True)
        robust (bool): if True plot with the robust method (see xarray documentation) (default = True)
        
    Returns:
        d_lbd_Si (real): wavelength shift
    """

    from .raman_hyperspectra_fit import fit_hyperspectra_Si
    from .raman_hyperspectra_baseline import flatten_hyperspectra

    db_rayleigh = db.sel(lbd=slice(db.lbd.values[0], lbd_end_flatten))
    db_rayleigh_flatten = flatten_hyperspectra(
        db_rayleigh, ("ials", 10_000, 0.01, 0.001)
    )  # ('linear',500,5)
    # ('als',10_000, 0.001,20)
    # ("top hat", par.factor)
    # ('ials',10_000,  0.01, 0.001)
    df_fit_rayleigh = fit_hyperspectra_Si(
        db_rayleigh_flatten,
        file_init,
        sheet="Rayleigh",
        lbd_deb_fit=lbd_deb_fit,
        lbd_end_fit=lbd_end_fit,
        save_xlsx=True,
    )

    lbd_0, d_lbd_Si, tol = fix_lbd0_py(
        0, df_fit_rayleigh, db.lbd.attrs["spectral_resolution"], tol_R2, 5, save=save
    )

    # plot the results
    R2_plot(
        df_fit_rayleigh,
        0,
        lbd_0,
        tol_R2,
        tol,
        lbd_deb_fit,
        lbd_end_fit,
        bins=bins,
        style=style,
        save=save,
        robust=robust,
    )

    return d_lbd_Si


def set_peak_to_value(
    db,
    lbd_si_c_1,
    file_init,
    sheet=None,
    tol_R2=0.99,
    lbd_deb_fit=480,
    lbd_end_fit=600,
    bins=50,
    style="seaborn",
    save=True,
    robust=True,
):
    """
    Wavelength shift correction using the location of a Raman peak
    
    Arguments:
        db (xarray): xarray containing the Rayleigh peak
        lbd_si_c_1 (real): location of the characteristic Raman peak 
        file_init (string): full path of the file containning the model
        sheet (string): name of the sheet (default = None)
        tol_R2 (real) : only the fit with R2> tol_R2 are used to compute the mean Rayleigh peak position (default =0.99)
        lbd_dep_fit (real): fitting lower wavelength in cm-1 (default = -40)
        lbd_end_fit (real): fitting upper wavelength in cm-1 (default = 40)
        lbd_end_flatten (real): upper wavelenght for the baseline removal in cm-1 (default = 70)
        bins (integer): number of bins for the histogramm
        style (string): plot style (default = 'seaborn')
        save (bool): if True saves the plot (default = True)
        robust (bool): if True plot with the robust method (see xarray documentation) (default = True)
        
    Returns:
        d_lbd_Si (real): wavelength shift in cm-1
    """

    from .raman_hyperspectra_fit import fit_hyperspectra_Si
    import numpy as np

    # fit the hyperspectra
    df_fit_Si = fit_hyperspectra_Si(
        db,
        file_init,
        sheet=sheet,
        lbd_deb_fit=lbd_deb_fit,
        lbd_end_fit=lbd_end_fit,
        save_xlsx=True,
    )

    # mean wavelength value of the cSi peak
    # lbd_0, d_lbd_Si, tol = rhp.fix_lbd_0(lbd_si_c_1, df_fit_Si, db, tol_R2, 30 , save = True)
    lbd_0, d_lbd_Si, tol = fix_lbd0_py(
        lbd_si_c_1,
        df_fit_Si,
        db.lbd.attrs["spectral_resolution"],
        tol_R2,
        30,
        save=True,
    )

    # plot the results
    R2_plot(
        df_fit_Si,
        lbd_si_c_1,
        lbd_0,
        tol_R2,
        tol,
        lbd_deb_fit,
        lbd_end_fit,
        bins=bins,
        style=style,
        save=save,
        robust=robust,
    )

    if np.isnan(lbd_0):
        raise Exception(f"R2 = {tol_R2} too high no spectrum found")

    return d_lbd_Si


def R2_plot(
    df,
    lbd_si_c_1,
    lbd_0,
    R_cSi,
    tol,
    lbd_deb_fit,
    lbd_end_fit,
    bins=50,
    style="seaborn",
    save=True,
    robust=True,
):

    """
    Plots the results obtained by set_rayleigh_peak_to_0 or by set_peak_to_value
    
    Arguments:
        df (data array):  contains the result of the fit rows : spectra, columns fitted param, R2
        lbd_si_c_1 (real): location of the characteristic Raman peak 
        R_cSi (real): threshold only spctra with R2> R_cSI are retained to compute the mean shift value
        tol: only spectra in the interval [lbd_si_c_1-tol, lbd_si_c_1-tol] are retained to compute the mean shift value
        tol_R2 (real) : only the fit with R2> tol_R2 are used to compute the mean Rayleigh peak position (default =0.99)
        lbd_dep_fit (real): fitting lower wavelength in cm-1 
        lbd_end_fit (real): fitting upper wavelength in cm-1 
        lbd_end_flatten (real): upper wavelenght for the baseline removal in cm-1 (default = 70)
        bins (integer): number of bins of the histogramm
        style (string): plot style (default = 'seaborn')
        save (bool): if True saves the plot (default = True)
        robust (bool): if True plot with the robust method (see xarray documentation) (default = True)
        
    Returns:
        None
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr
    import sys
    import seaborn as sns
    import time

    def plot_R2():

        fig = plt.figure(figsize=(10, 10))
        plt.style.use(style)
        plt.tight_layout()

        fig.suptitle(
            f'Fitting function : {df.columns[0].split(":")[0].strip()[:-1]}',
            fontsize=16,
        )

        plt.subplot(2, 2, 1)
        plt.hist(df["R2"], bins=bins, edgecolor="black")
        plt.title("R2 histogramme ")
        plt.xlabel("R2")

        plt.subplot(2, 2, 2)
        da_fit["R2"].plot(robust=robust)
        plt.xlabel("Index_x")
        plt.ylabel("Index_y")
        x = da_fit["R2"].values.flatten()
        max_R2 = np.max(x[~np.isnan(x)])
        plt.title(f"max R2 = {round(max_R2,5)}")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.subplots_adjust(wspace=0.5, hspace=0.3)

        plt.subplot(2, 2, 3)
        idx_Lbd_0 = [idx for idx, x in enumerate(df.columns) if "Lbd_0" in x][0]
        sns.scatterplot(df.columns[idx_Lbd_0], "R2", s=15, marker="8", data=df)
        plt.title(f"Delta_lbd [cm-1] = {round(lbd_0 - lbd_si_c_1,2)}")
        plt.xlabel("Lbd_0 [cm-1] ")
        plt.plot(
            [lbd_si_c_1, lbd_si_c_1],
            [0.1, 1],
            "--k",
            label=f"lbd_si_c_1 : {lbd_si_c_1 :.2f} cm-1",
        )
        plt.plot([lbd_0, lbd_0], [0.1, 1], "--r", label=f"lbd_0 : {lbd_0 :.2f} cm-1")
        plt.legend()

        plt.subplot(2, 2, 4)
        name_col_lbd_0 = [x for x in df.columns if "Lbd_0" in x][0]
        x = da_fit[name_col_lbd_0].values.flatten()
        max_lbd_0 = np.max(x[~np.isnan(x)])
        min_lbd_0 = np.min(x[~np.isnan(x)])
        plt.scatter(df[name_col_lbd_0], df["R2"], s=5)
        plt.plot([lbd_si_c_1 - tol, lbd_si_c_1 + tol], [R_cSi, R_cSi], "--r")
        plt.plot([lbd_si_c_1 - tol, lbd_si_c_1 + tol], [1, 1], "--r")
        plt.plot([lbd_si_c_1 - tol, lbd_si_c_1 - tol], [R_cSi, 1], "--r")
        plt.plot([lbd_si_c_1 + tol, lbd_si_c_1 + tol], [R_cSi, 1], "--r")
        plt.ylabel("R2")
        plt.xlabel("Lbd0 [cm-1]")
        plt.ylim((0.9, 1.01))
        plt.title(f"R2 limit = {R_cSi}, tol= {tol} [cm-1]")
        if save:
            fig = plt.gcf()
            fig.savefig(
                sys._getframe().f_code.co_name
                + " "
                + time.strftime("%H_%M_%S")
                + ".png"
            )
        plt.show()

    def plot_lbd_0():

        fig = plt.figure(figsize=(10, 5))
        plt.style.use(style)
        plt.tight_layout()

        plt.subplot(1, 2, 1)
        name_col_lbd_0 = [x for x in df.columns if "Lbd_0" in x][0]
        plt.hist(np.array(df[name_col_lbd_0]), bins=bins, edgecolor="black")
        plt.title(f"{lbd_deb_fit} < lbd-fit [cm-1] < {lbd_end_fit}")
        plt.xlabel("Lbd_0 [cm-1] ")

        plt.subplot(1, 2, 2)
        x = da_fit[name_col_lbd_0].values.flatten()
        max_lbd_0 = np.max(x[~np.isnan(x)])
        min_lbd_0 = np.min(x[~np.isnan(x)])
        da_fit[name_col_lbd_0].plot(robust=robust)
        plt.xlabel("Index_x")
        plt.ylabel("Index_y")
        plt.title(f"{round(min_lbd_0,0)} < Lbd_0 [cm-1] < {round(max_lbd_0,0)}")
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

    da_fit = xr.Dataset.from_dataframe(df)
    plot_R2()
    plot_lbd_0()

    plt.style.use("default")


def update_xarray(Y, file_info, L_x_default, L_y_default, d_lbd_Si=0):

    """
    Affects physical values to x and y axes and shift the vawelength to satisfie lbd0 = lbd_si_c_1
    
    Arguments:
        Y (list of xarray): xarrays be uptaded
        file_info (string): information Witec file
        L_y_default (real): default y length in microns
        L_x_default (real): default x length in microns
        d_lbd_si (real): shitf of the vawelength in cm-1
    """

    import numpy as np
    from raman_hyperspectra.raman_hyperspectra_read_files import (
        read_RAMAN_WITEC_information,
    )

    #if not (isinstance(Y, list)):
    #    raise NameError("type error in update_xarray: Y not a list ")
        
    #Y = np.atleast_1d(Y) # generate a bug to be understood

    Z = []
    if Y[0].attrs["tool"] == "WITEK":

        # get the physical size of the images
        try:

            dic_info = read_RAMAN_WITEC_information(file_info)
            L_x = dic_info["Scan Width [µm]"]
            L_y = dic_info["Scan Height [µm]"]

        except:

            L_x = L_x_default
            L_y = L_y_default

        for X in Y:

            X = X.assign_coords({"x": (X.x * L_x / (X.x.size - 1))})
            X.x.attrs["units"] = "µm"
            X = X.assign_coords({"y": (X.y * L_y / (X.y.size - 1))})
            X.y.attrs["units"] = "µm"

            try:  # the xarray has lbd coordonates

                lbd_attrs = X.lbd.attrs
                X = X.assign_coords({"lbd": (X.lbd.values - d_lbd_Si)})
                X.lbd.attrs = lbd_attrs
                X.lbd.attrs["shift correction"] = "yes"

            except:
                pass

            Z.append(X)

    elif Y[0].attrs["tool"] == "RENISHAW":

        for X in Y:

            try:  # the xarray has lbd coordonates
                lbd_attrs = X.lbd.attrs
                X = X.assign_coords({"lbd": (X.lbd.values - d_lbd_Si)})
                X.lbd.attrs = lbd_attrs
                X.lbd.attrs["shift correction"] = "yes"

            except:
                pass

            Z.append(X)
    else:

        raise ValueError("unknown tool must be WITEK or RENISHAW")

    return Z


def fix_lbd0_py(lbd_si_c_1, df, spectral_resolution, R_cSi, tol_default, save=True):

    import tkinter as tk
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg,
        NavigationToolbar2Tk,
    )
    from matplotlib.figure import Figure
    from matplotlib.backend_bases import key_press_handler
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib

    lbd_0_f = lambda Lbd_0_min, Lbd_0_max, R_cSi: dg.query(
        "R2 >= @R_cSi & Lbd_0 > @Lbd_0_min & Lbd_0 < @Lbd_0_max"
    )["Lbd_0"].mean()

    dg = df.copy()
    # change the column names to eliminate the function name
    # ex Pseudo_Voigt0 : Lbd_0 ---> Lbd_0
    dg.columns = [x.split(":")[1].strip() if ":" in x else x for x in df.columns]

    def plot():
        tol_max = 200
        lbd_0_default = lbd_0_f(
            lbd_si_c_1 - tol_default, lbd_si_c_1 + tol_default, R_cSi
        )
        x = list(range(1, tol_max, 1))
        y = [lbd_0_f(lbd_si_c_1 - tols, lbd_si_c_1 + tols, R_cSi) for tols in x]

        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(
            x, np.array(y) - lbd_si_c_1, tol_default, lbd_0_default - lbd_si_c_1, "or"
        )
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))
        ax.set_xlabel("tol [cm-1] ")
        ax.set_ylabel("delta ldb_0 [cm-1]")
        ax.set_title(f"spectral resolution : {round(spectral_resolution ,2) } cm-1")

        canvas = FigureCanvasTkAgg(fig, master=frame_plot)
        canvas.draw()
        canvas.get_tk_widget()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, frame_plot)  # set ul the plot tool bar
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def get_tol():
        global tol_g, lbd_0
        get_tol.flag = getattr(get_tol, "flag", True)
        tol = entry.get()
        tol = float(tol)
        tol_g = tol

        lbd_0 = lbd_0_f(lbd_si_c_1 - tol, lbd_si_c_1 + tol, R_cSi)

        myLabel = tk.Label(frame_input, text=f"  lbd_0 : {str(round(lbd_0,2))} cm-1")
        myLabel.grid(row=3, column=0)

        if get_tol.flag:
            button = tk.Button(master=frame_input, text="quit", command=quit)
            button.grid(row=2, column=2)
            get_tol.flag = False

    def quit():
        root.quit()
        root.destroy()

    root = tk.Tk()

    root.wm_title("Embedding in tkinter")
    frame_plot = tk.Frame(root)
    frame_input = tk.Frame(root)
    frame_plot.grid(row=0, column=0)
    frame_input.grid(row=1, column=0)

    plot()
    entry = tk.Entry(frame_input)
    entry.grid(row=2, column=1)
    define_tol_button = tk.Button(
        frame_input, text="Enter tolerance [cm-1]  : ", command=get_tol
    )
    define_tol_button.grid(row=2, column=0)

    tk.mainloop()

    return lbd_0, lbd_0 - lbd_si_c_1, tol_g
