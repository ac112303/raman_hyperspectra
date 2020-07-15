"""
Computation of the number of occurrences, versus the wavelength, of a Raman peak in an hyperspectrum.

3rd party dependencies : 
numpy, scipy, pandas, djustText, matplotlib, tqdm 

"""

__all__ = ["find_MAX", "Stat_MAX", "zoom_Stat_MAX", "phase_maximums_imaging"]


def find_MAX(z, Ecart=5, prominence=0.5, height=3):

    """
    Buil the list of the indice of the Raman peak location
    
    Arguments:
           z (np array): spectrum intensities
           Ecart (float): If the distance between two consecutive peaks height is lesser than Ecart
                  than the only the higher peak is retained
           height (float): to be selected th height of a peak must be greater than height
           prominence (float): to be selected the prominence of a peak must be lesser than prominence
           
    Returns:
            idx_max (list of int) : list of the peal maximum index
            zd (np array) : list of the peaks height
    """

    # 3rd party dependencies
    from scipy.signal import find_peaks
    import numpy as np

    idx_max_tot, info_peak = find_peaks(
        z, distance=Ecart, prominence=prominence, height=height
    )

    idx_max = list(idx_max_tot)

    return idx_max


def Stat_MAX(
    da,
    threshold_peak=100,
    Ecart=5,
    prominence=0.5,
    height=3,
    dic_lbds_ref=None,
    dic_class_Si=None,
    lbd_min_plot=None,
    lbd_max_plot=None,
    save=True,
    show=True,
):
    """
    
    Arguments:
        da (xarray): hyperspectra
        threshold_peak (int) : only the cumulative with a height greater than threshold_peak are selected
        Ecart (float): If the difference of two consecutive peaks sheight is lesser than Ecart
                  than the only the higher peak is retained
        prominence (float): to be selected the prominence of a peak must be lesser than prominence
        height (float): to be selected th height of a peak must be greater than height
        dic_lbds_ref (dict): {location (cm-1):phase name,... } use for graphical representation (default = None)
        dic_class_Si (dict): {phase name: [lbd min (cm-1), lbd_max (cm-1)],...}
                             [lbd min (cm-1), lbd_max (cm-1)] are the wavelength extends of the phase.
        lbd_min_plot: the lesser wavelength used for the plot (default = None)
        lbd_max_plot: the greater wavelength used for the plot (default = None)
        save (bool) : if True the plot is save
        show (bool): if True the plot is shown
    
    Returns:
        df (DataFrame): lbd_max|peak height|peak width
    """

    # Standard Library dependencies
    import bisect
    import itertools as it
    import sys
    import time

    # 3rd party dependencies
    import numpy as np
    from scipy.signal import find_peaks, peak_widths
    import pandas as pd
    from adjustText import adjust_text
    import matplotlib.pyplot as plt

    relative_height = 0.5  # relative heigth for the peak width determination

    X = da.data.reshape((da.shape[0] * da.shape[1], da.shape[2]))
    lbd = da.lbd.data

    # find the indice of lbd_min_plot and lbd_max_plot

    if lbd_min_plot is None:
        lbd_min_plot_ = lbd[0]
        idx_lbd_min_plot_ = 0
    else:
        lbd_min_plot_ = lbd_min_plot
        idx_lbd_min_plot_ = bisect.bisect_right(lbd, lbd_min_plot)

    if lbd_max_plot is None:
        lbd_max_plot_ = lbd[-1]
        idx_lbd_max_plot_ = len(lbd) - 1
    else:
        lbd_max_plot_ = lbd_max_plot
        idx_lbd_max_plot_ = bisect.bisect_right(lbd, lbd_max_plot)

    # build the list idx_peak of the indice of ALL the maxima of the hyperspectra
    idx_peak = []
    for i in range(da.shape[0] * da.shape[1]):
        idx_max = find_MAX(X[i, :], Ecart=5, prominence=0.5, height=3)
        idx_peak = idx_peak + idx_max

    # build the number of occurrences of a peak vs the wavelength index
    pos_stat = np.bincount(np.array(idx_peak))
    pos_stat = np.concatenate(
        (pos_stat, np.zeros(len(lbd) - len(pos_stat))), axis=None
    )  # add leading zeros to match
    # the size of pos_stat and lbd
    # compute the list "peak occurrence" width and the list of peak height
    idx_peak_max, info_peak = find_peaks(pos_stat, height=threshold_peak)
    peak_half = peak_widths(pos_stat, idx_peak_max, rel_height=relative_height)
    peak_half = peak_half[0]

    peak_heights = info_peak["peak_heights"]
    lbd_max = lbd[idx_peak_max]

    # build the dataframe ordered by decreasing peak heights
    sorted_lbd_max = sorted(range(len(lbd_max)), key=lambda k: lbd_max[k])
    dict_max = {
        idx: [tup[0], tup[1], tup[2], tup[3]]
        for idx, tup in enumerate(
            sorted(
                list(zip(sorted_lbd_max, lbd_max, peak_heights, peak_half)),
                key=lambda tup: tup[2],
                reverse=True,
            )
        )
    }
    df = pd.DataFrame.from_dict(
        dict_max,
        orient="index",
        columns=[
            "Indice",
            "lbd_max",
            "peak height",
            "peak width " + str(relative_height),
        ],
    )

    # plotting

    fig = plt.figure(figsize=(20, 10))
    plt.plot(lbd, pos_stat)
    plt.plot(lbd_max, pos_stat[idx_peak_max], "ob")
    plt.plot([lbd[0], lbd[-1]], [threshold_peak, threshold_peak], "g")
    plt.xlabel("ldb [cm-1]")
    bottom = -max(pos_stat[idx_lbd_min_plot_:idx_lbd_max_plot_]) / 8
    top = 1.2 * max(pos_stat[idx_lbd_min_plot_:idx_lbd_max_plot_])
    plt.ylim(bottom=bottom)
    plt.ylim(top=top)
    plt.xlim(lbd_min_plot_, lbd_max_plot_)

    # tracé des indices des pics en position et leur indice de classement par hauteur décroissante

    texts = []  # indices du peak en postion et en hauteu
    texts1 = []  # labels des phases identifiées
    for idx, x in enumerate(df.values):

        if (x[1] > lbd_min_plot_) & (x[1] < lbd_max_plot_):

            texts.append(
                plt.text(x[1], x[2], str(int(x[0])) + ", " + str(int(idx)), size=9)
            )

            texts1.append(
                plt.text(x[1], bottom / 2, str(int(x[1])), size=9, rotation=90)
            )

    adjust_text(texts, save_steps=False)
    adjust_text(texts1, save_steps=False)

    # tracé des labels et des positions tabulées des phases identifiées

    if dic_lbds_ref is not None:  # labels of the tabulated phases in dic_lbds_ref dict

        lbd_ = np.sort(np.array(df["lbd_max"]))
        texts2 = []
        for x in zip(dic_lbds_ref.keys(), dic_lbds_ref.values(), it.cycle("rbkygmc")):

            if (x[0] > lbd_min_plot_) & (x[0] < lbd_max_plot_):

                idx = np.argmin(np.abs(lbd_ - x[0]))
                texts2.append(
                    plt.text(
                        x[0],
                        peak_heights[idx] - 1.5 * bottom,
                        f"{x[1]} ({str(x[0])})",
                        size=9,
                        rotation=90,
                        horizontalalignment="center",
                        verticalalignment="top",
                        color=x[2],
                    )
                )

        adjust_text(texts2, save_steps=False)

        # plo the phase extend

    if dic_class_Si is not None:

        for x in zip(dic_class_Si.values(), it.cycle("rbkygmc"), it.cycle(["-", "--"])):
            col = x[1]
            line_type = x[2]
            plt.plot([x[0][0], x[0][0]], [0, top], col + line_type)
            plt.plot([x[0][1], x[0][1]], [0, top], col + line_type)

    Stat_MAX.idx = getattr(Stat_MAX, "idx", 0)
    if save:
        fig = plt.gcf()
        Stat_MAX.idx += 1
        fig.savefig(
            sys._getframe().f_code.co_name
            + " "
            + str(Stat_MAX.idx)
            + " "
            + time.strftime("%H_%M_%S")
            + ".png"
        )

    if show:
        plt.show()
    else:
        plt.close()

    Stat_MAX.flag = getattr(Stat_MAX, "flag", True)
    if Stat_MAX.flag:  # First call
        df.sort_values(by=["lbd_max"], inplace=True)
        df.to_excel("histo_maximums.xlsx")
        Stat_MAX.flag = False

    return df


def zoom_Stat_MAX(
    da,
    threshold_peak=100,
    Ecart=5,
    prominence=0.5,
    height=3,
    dic_lbds_ref=None,
    dic_class_Si=None,
    lbd_min_plot=None,
    lbd_max_plot=None,
    save=True,
    show=False,
):

    import raman_hyperspectra as rhp
    from tqdm import trange

    for idx in trange(len(lbd_min_plot), desc="zoom max"):
        dg = Stat_MAX(
            da,
            threshold_peak=threshold_peak,
            dic_lbds_ref=dic_lbds_ref,
            dic_class_Si=dic_class_Si,
            lbd_min_plot=lbd_min_plot[idx],
            lbd_max_plot=lbd_max_plot[idx],
            save=save,
            show=show,
        )
    return dg


def phase_maximums_imaging(
    da,
    phase,
    class_Si,
    Ecart=5,
    prominence=0.5,
    height=3,
    Norme=True,
    robust=True,
    save=True,
    cmap="viridis",
    path_image=None,
):

    """
    phase imaging
    
    Arguments:
        da : xarray containing the phases maxumum vs x,y
        phase : list of the phases to be imaged
        class_si : dictionary {classe label : [list of phases pertening to the classe]}
        Norme : if True the amplitude are normalizez vs the maximum amplitude of the hyperspectrum
        robust : if True the amplitude of the two last centiles are trimmed
        save : if True the plot is save
        
    Returns:
        db : xarrays of phase images
        dic_phase_max : {phase: [idx_x max, idx y max, lbd_max,phase peak height] }
    """

    # Standard libary dependencies
    import bisect
    import sys
    import time

    # 3rd party dependencies
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt

    X = da.data
    lbd = da.lbd.data

    phase = list(set(phase))  # eliminate duplicated phase

    bins = []  # bins contruction
    for x in phase:
        bins.append(class_Si[x])

    phase_ = []
    for x in sorted(list(zip(bins, phase)), key=lambda tup: tup[0][0], reverse=False):
        phase_.append(x[1])

    bins.append([lbd[0], lbd[-1]])
    bins = sorted(sum(bins, []))

    Y = np.zeros(
        (X.shape[0], X.shape[1], len(phase))
    )  # Y will contain the peak intensity vs x, y, phase
    res = [[] for i in range(len(phase))]

    for i in range(X.shape[0]):

        for j in range(X.shape[1]):

            idxs_max = find_MAX(
                X[i, j, :], Ecart=5, prominence=0.5, height=3
            )  # search all the indices of the maximums
            lbd_max = lbd[idxs_max]

            idxs_bin = np.digitize(lbd_max, bins, right=True)  # sifts peaks into bins

            for num_phase in range(1, len(phase_) + 1):  # allocate maximums to phase

                idxs_bin_class = np.where(idxs_bin == 2 * num_phase)[
                    0
                ]  # select indices belonging to a phase

                if idxs_bin_class.size > 0:  # more than one maximum belong to the class
                    list_peak_max = []
                    list_idx = []

                    for (
                        idx_bin_class
                    ) in (
                        idxs_bin_class
                    ):  # list of index and peak intensity of maximums belonging to a class

                        idx_ = bisect.bisect(lbd, lbd_max[idx_bin_class]) - 1
                        list_idx.append(idx_)
                        list_peak_max.append(X[i, j, idx_])

                    # if more than two peaks are in a bin we select the one with the highest intensity
                    peak_max, idx = sorted(
                        list(zip(list_peak_max, list_idx)),
                        key=lambda tup: tup[0],
                        reverse=True,
                    )[0]

                    Y[i, j, num_phase - 1] = peak_max
                    res[num_phase - 1].append((i, j, lbd[idx], peak_max))

    if Norme:  # normalization by the maximum intensity
        Y_max = np.max(Y)
        for i in range(len(phase)):
            Y_phase_max = np.max(Y[:, :, i])
            try:
                ratio = Y_max / Y_phase_max
                Y[:, :, i] = ratio * Y[:, :, i]
                ratio = int(ratio)
                phase_[i] = phase_[i] + f"(X{ratio})"
            except:  # zero image
                pass

    dic_phase_max = {x[0]: x[1] for x in zip(phase_, res)}

    x = da.x.values
    y = da.y.values
    db = xr.DataArray(
        Y.reshape((X.shape[0], X.shape[1], len(phase_))),
        dims=["y", "x", "phase"],
        name="Intensity",
        attrs={"units": "u.a."},
        coords={
            "y": xr.DataArray(
                y, name="y", dims=["y"], attrs={"units": da.y.attrs["units"]}
            ),
            "x": xr.DataArray(
                x, name="x", dims=["x"], attrs={"units": da.x.attrs["units"]}
            ),
            "phase": xr.DataArray(
                phase_, name="phase", dims=["phase"], attrs={"units": ""}
            ),
        },
    )

    if path_image:
        print(path_image)
        db = add_image2xarray(db, path_image)

    g_simple = db.plot(x="x", y="y", col="phase", robust=robust, col_wrap=3, cmap=cmap)

    if save:
        fig = plt.gcf()
        fig.savefig(
            sys._getframe().f_code.co_name + " " + time.strftime("%H_%M_%S") + ".png"
        )

    plt.show()

    return db, dic_phase_max
