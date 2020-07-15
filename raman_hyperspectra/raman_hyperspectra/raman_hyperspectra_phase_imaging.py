"""
Imaging of the the phase. The z value is the peak height. 
Before executing these function you must:
    - remove the cosmics (not mandatory)
    - remove the baseline
    - denoise the spectra to eliminate spurious maxima

Internal functions
 add_image2xarray
 get_pixel_location

3rd party dependencies : 
xarray, numpy, matplotlib, PIL, adjustText, pandas, OpenCV
"""

__all__ = [
    "choose_phase_xarray",
    "menu_phase",
    "phase_maximums_imaging",
    "classe_phase_maximums_imaging",
    "interactive_plot",
]


def classe_phase_maximums_imaging(
    da,
    class_Si,
    dic_ref,
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
    Build the image of the peak intensity of a phase.
    For matter of convenience the phase are divided into classes (The name of classe is arbitray).
    The function produce one image per classe. If if present the topographical image will be plotted 
    together with the phases image.
    
    Arguments:
        da (DataArray): hyperspectrum
        class_si (dict of dict): contain the limits of the bins  {classe: {classe member : [lbd_min, bd_max]...}...}
                       ex : {'A': {'si_a_3': [130, 160], 'si_c_2': [285, 315],  'si_a_1': [450, 480],  'si_c_1': [516, 526]},...}
        dic_ref (dic of dict): contain the referenced wavelength {classe: {classe member : lbd_ref.}
                       ex: {'A': {145: 'si_a_3', 300: 'si_c_2', 465: 'si_a_1', 521: 'si_c_1'},
        Norme (bool): if True all the maxima of the phase are normalized versus the maximum of the hyperspectra
        robust (bool): if True the two last centiles on the image are trimmed (see the xarray documentation)
        save (bool): if True we save the image
        cmap (string): colormap (default = 'viridis'
        path_image (string): the full path of the topographical image cropped to the size of the ROI. (default = None)
                             
        
    Returns:
        dic_image_phase (dict): {classe : DataArray of phase images}
        dic_dic_phase_max (dict): dictionary of dictionaries
                            {classe : {phase: [idx_x max, idx y max, lbd_max,height, ] }}
        df_lbd (DataFrame): data frame  phase name | lbd [cm-1] |std [cm-1] | % occurrence
    """

    # Standard Library dependencies
    from functools import reduce

    # 3rd party dependencies
    import pandas as pd
    import numpy as np

    # extraction de la phsase si_c_1 qui va servir de référence pour normer l'intensité des pics
    class_si_c1 = {
        "si_c_1": reduce(
            lambda x, y: dict(**x, **y), list([X for _, X in class_Si.items()])
        )["si_c_1"]
    }

    dic_image_phase = {}  # dictionnaire contenant les xarrays générés
    dic_dic_phase_max = {}

    for X in dic_ref.keys():  # iteration over classes

        class_ = {**class_Si[X], **class_si_c1}
        phase = list(dic_ref[X].values()) + ["si_c_1"]  # add phase si_c_1

        db, dic_phase_max = phase_maximums_imaging(
            da,
            phase,
            class_,
            Ecart,
            prominence,
            height,
            Norme,
            robust,
            save,
            cmap=cmap,
            path_image=path_image,
        )
        dic_image_phase[X] = db
        dic_dic_phase_max[X] = dic_phase_max

    # Build a data frame : |phase name | lbd [cm-1] |std [cm-1] | % occurrence |

    dic_lbd_phase_exp = {}
    for X in dic_dic_phase_max.keys():
        for phase, value in dic_dic_phase_max[X].items():
            dic_lbd_phase_exp[phase.split("(")[0]] = [
                np.mean([X[2] for X in value]),
                np.std([X[2] for X in value]),
                100 * len(value) / (da.shape[0] * da.shape[1]),
            ]
    df_lbd = pd.DataFrame.from_dict(dic_lbd_phase_exp).T
    df_lbd.columns = ["lbd [cm-1]", "std [cm-1]", "% occurrence"]
    df_lbd.to_excel("lbd_phase_exp.xlsx")

    return dic_image_phase, dic_dic_phase_max, df_lbd


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
    phase imgaging of a classe of phases
    
    Arguments:
        da (DataArray):  containing the phases maxumum vs x,y
        phase (list of strings): list of the phases to be imaged
        class_si (dict):  {label of th classe : [list of phases pertening to the classe]}
        Norme (bool): if True the amplitude are normalizez vs the maximum amplitude of the hyperspectrum
        robust (bool): if True the amplitude of the two last centiles are trimmed
        save (bool): if True the plot is save
        
    Returns:
        db (DataArray):  a data array containing phase images (coord "phase")
        dic_phase_max (dict of dicts): {phase: [idx_x max, idx y max, lbd_max,phase peak height] }
    """

    # Standard Library dependencies:
    import bisect
    import sys
    import time

    # Internal mole dependencies
    from raman_hyperspectra.raman_hyperspectra_stat_max import find_MAX

    # 3rd party dependencies
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt

    X = da.data
    lbd = da.lbd.data

    phase = list(set(phase))  # eliminate duplicated phase
    bins = []  # construct bins
    for x in phase:
        bins.append(class_Si[x])

    phase_ = []
    for x in sorted(list(zip(bins, phase)), key=lambda tup: tup[0][0], reverse=False):
        phase_.append(x[1])

    bins.append([lbd[0], lbd[-1]])
    bins = sorted(sum(bins, []))

    Y = np.zeros((X.shape[0], X.shape[1], len(phase)))
    res = [[] for i in range(len(phase))]

    for i in range(X.shape[0]):

        for j in range(X.shape[1]):

            idxs_max = find_MAX(
                X[i, j, :], Ecart=5, prominence=0.5, height=3
            )  # search all the indices of the maximums
            lbd_max = lbd[idxs_max]

            idxs_bin = np.digitize(
                lbd_max, bins, right=True
            )  # allocate to each peaks the index i of the interval ]bin(i-1),bin(i)]
            # to which it belongs

            for num_phase in range(1, len(phase_) + 1):  # allocate maximums to a phase

                idxs_bin_class = np.where(idxs_bin == 2 * num_phase)[
                    0
                ]  # select the indices of peaks belonging to the num_phase phase

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

    if Norme:  # normalization by the maximum intensity of the classe
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

    x = da.x.values  # create xarray
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

    if path_image:  # if present a topographical image is added to the stack
        db = add_image2xarray(db, path_image)

    g_simple = db.plot(x="x", y="y", col="phase", robust=robust, col_wrap=3, cmap=cmap)

    if save:
        fig = plt.gcf()
        fig.savefig(
            sys._getframe().f_code.co_name + " " + time.strftime("%H_%M_%S") + ".png"
        )

    plt.show()

    return db, dic_phase_max


def add_image2xarray(da, path_image, flip=True):

    """
    add a topographic image to an xarray
    
    Arguments:
        da (DataArray): xarray to which the image will be added
        path_image (string): the absolute path to the image
        
    Returns:
        da_img (xarray):  concatenation of da an the image
    """

    # 3rd party dependencies
    from PIL import Image
    import numpy as np
    import xarray as xr

    im = Image.open(path_image).convert("LA")
    im = im.resize((da.shape[0], da.shape[1]))
    xx = (
        np.array(im).flatten()[::2].reshape((da.shape[0], da.shape[1]))
    )  # supress the second plane of the image
    if flip:
        xx = xx[::-1, :]  # flip the image upside down
    max_phase = np.max(da).values
    Image = np.expand_dims(xx, axis=2) * max_phase / np.max(xx) / 3

    x = da.x.values
    y = da.y.values
    db = xr.DataArray(
        Image,
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
                ["Image"], name="phase", dims=["phase"], attrs={"units": ""}
            ),
        },
    )
    da_img = xr.concat([da, db], "phase")

    return da_img


def choose_phase_xarray(list_phase, dic_image_phase, dic_dic_phase_max):

    """
    picks an phase xarray from dic_image_phase generated by classe_phase_maximums_imaging
    
    Arguments:
        df_lbd (data frame):    phase name | mean vawelength
        dic_image_phase (dict of xarrays):  {classe : xarrays of phase images}
        dic_dic_phase_max (dict of dict): {classe : {phase: [idx_x max, idx y max, lbd_max,height, ] }}
    Returns:
        da (xarray): selected phase image

    """

    name_phase = menu_phase(list_phase)  # choose the phase

    if (len(name_phase) == 0) or ("no phase" in name_phase):  # for future exit
        da = None

    else:
        name_phase = name_phase[0]

        # recherche de la classe à laquelle appartient name_phase
        dic_phase = {}
        for (
            X
        ) in (
            dic_dic_phase_max.keys()
        ):  # get rid of the magnification "(Xnnn)" of the phase name
            dic_phase[X] = [x.split("(")[0] for x in list(dic_dic_phase_max[X].keys())]

        for Y in dic_dic_phase_max.keys():
            if name_phase in dic_phase[Y]:
                classe = Y
                break

        # build the phase image
        name_phase_ext = [
            X for X in dic_image_phase[Y].phase.values if name_phase in X.split("(")[0]
        ]
        da = dic_image_phase[Y].sel(phase=name_phase_ext)

    return da


def interactive_plot(da, da_spectra, path_image=None, percentile=0.98, save=True):

    """
    plot of the spectrum which spatial location interactively on the image of da
    
    Arguments:
        da (DataArray): xarray from image of which we pick the spatial coordinates of the spectrum of interest
        da_spectra (DataArray): xarray from which the spectra are plotted
        path_image (string): absolute path of the topographical image of the sample
        percentile (float): trim the spectra pertening to th 100-percentle last percentiles
        save (bool): if True the image is saved
        
    Returns:
        dpick (DataFrame): of the selected point label | index x | index y
    """

    # Standard Library dependencies
    import itertools as it
    import sys
    import string
    import time

    # 3rd party dependencies
    import numpy as np
    import pandas as pd
    from PIL import Image
    from adjustText import adjust_text
    import matplotlib.pyplot as plt

    interactive_plot.idx_call = getattr(
        interactive_plot, "idx_call", 0
    )  # mimic a static variable
    interactive_plot.idx_call += 1

    L_x = da.x.values[-1]  # size of the image [µm]
    L_y = da.y.values[-1]

    if len(da.shape) == 2:
        A = np.array(da.values)

    elif (len(da.shape) == 3) & (da.shape[2] == 1):  # not brillant coding
        A = da.values.reshape((da.shape[0], da.shape[1]))

    elif (len(da.shape) == 3) & (da.shape[0] == 1):
        A = np.array(da.values)[0]

    n_row, n_col = np.shape(A)
    img = A[::-1, :] * 255 / np.max(A)  # <-- normalisation + tricky x reflection
    img[np.where(img > np.percentile(img, percentile))] = np.percentile(
        img, 98
    )  # trim the 2 last centiles

    plt.subplot(2, 2, 2)

    # plot and store the phase intensity image to be used by cv2
    plt.imshow(img, cmap="viridis")
    plt.axis("off")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(
        r"c:\Temp\real_and_recon.tiff", pad_inches=0, bbox_inches="tight"
    )  # ! don't cchange these parameters !!

    # pick pixel in the phase image
    x, y, n_row_img, n_col_img = get_pixel_location("c:\\Temp\\real_and_recon.tiff")

    # plot result
    plt.close()
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 2)
    da.plot(robust=True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("on")
    plt.scatter(x * L_x / n_row_img, L_y - y * L_y / n_col_img, s=2, c="r", marker="s")

    # plot results
    # plot topographic image
    plt.subplot(2, 2, 1)
    if path_image:
        img_topo = Image.open(path_image)  # plt.imread(path_image)
        img_topo = img_topo.resize((da.shape[0], da.shape[1]))
        (
            n_row_topo,
            n_col_topo,
        ) = img_topo.size  # n_row_topo,n_col_topo, _ = img_topo.shape
    else:
        img_topo = np.zeros((da_spectra.shape[0:2]))
        n_row_topo, n_col_topo = 50, 50

    plt.imshow(img_topo, extent=[0, L_x, 0, L_y])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim((0, L_x))
    plt.ylim((0, L_y))
    plt.grid(b=None)

    # plot selected point and label on the topographic image
    text = []
    for X in zip(x, y, it.cycle(string.ascii_lowercase)):
        plt.scatter(
            X[0] * L_x / n_row_img,
            L_y - X[1] * L_y / n_col_img,
            s=10,
            c="r",
            marker="s",
        )
        text.append(
            plt.text(
                int(X[0] * L_x / n_row_img),
                L_y - int(X[1] * L_y / n_col_img) + 3,
                X[2],
                color="r",
                size=8,
                horizontalalignment="center",
                verticalalignment="top",
            )
        )

    adjust_text(text)

    plt.subplot(2, 2, 3)  # plot the selected spectra

    dic_pick = {
        X[2]: [
            int(X[0] * (n_row) / n_row_img),
            n_col - int(X[1] * (n_col) / n_col_img) - 1,
        ]
        for X in zip(x, y, it.cycle(string.ascii_lowercase))
    }

    high_max = 0
    for label, index in dic_pick.items():
        t = da_spectra.isel(x=index[0], y=index[1])
        high_max = max(high_max, np.max(da_spectra.isel(x=index[0], y=index[1])))
        t.plot(label=label)
        plt.title("")
        plt.legend(loc="right", bbox_to_anchor=(0.95, 0.5))

    plt.subplot(2, 2, 4)  # plot the zoom of the selected spectra
    for label, index in dic_pick.items():
        t = da_spectra.isel(x=index[0], y=index[1])
        t.plot(label=label)
        plt.ylabel = ""
        plt.title("")
        # plt.legend(loc ='right', bbox_to_anchor=(0.9, 0.5))
    plt.ylim(0, high_max / 3)

    plt.subplots_adjust(wspace=0.2, hspace=0.1)

    if save:
        fig = plt.gcf()
        fig.savefig(
            sys._getframe().f_code.co_name + " " + time.strftime("%H_%M_%S") + ".png",
            bbox_inches="tight",
            pad_inches=0,
        )

    plt.show()

    dpick = pd.DataFrame(dic_pick).T
    dpick.columns = ["idx_x", "idx_y"]

    return dpick


def get_pixel_location(path_image, magnification_factor: "int" = 4):

    """
    interactive selection of pixels on an image
    
        Arguments::
            path_image (string): the absolute path of the image
            magnification_factor (int): magnification factor of the imagege
            
        Returns:
            n_col (int): nparray of the selected pixels columns (x)
            n_row (int): nparray of the selected pixels rows (y)
            n_row_img (int): number of  rows of the image
            n_col_img (int): number of columns of the image
    """

    # 3rd party dependencies
    import cv2
    import numpy as np

    xy = []

    def click_event(event, x, y, flags, param):
        if (event == cv2.EVENT_LBUTTONDOWN) or (event == cv2.EVENT_RBUTTONDOWN):
            cv2.rectangle(
                img,
                (x, y),
                (x - magnification_factor, y - magnification_factor),
                (255, 255, 255),
                2,
            )
            cv2.imshow("image", img)
            n_col = int(np.ceil(x / magnification_factor))
            n_row = int(np.ceil(y / magnification_factor))
            xy.append([n_col, n_row])

    img = cv2.imread(path_image)

    n_row_img, n_col_img, _ = img.shape
    img = cv2.resize(
        img,
        (
            round(n_col_img) * magnification_factor,
            round(n_row_img) * magnification_factor,
        ),
    )  # , interpolation=cv2.INTER_AREA)
    n_row_img_mag, n_col_img_mag, _ = img.shape
    cv2.imshow("image", img)

    cv2.setMouseCallback("image", click_event)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    xy = np.array(xy).flatten()
    n_col = np.array(xy[0::2])
    n_row = np.array(xy[1::2])

    return n_col, n_row, n_row_img, n_col_img


def menu_phase(list_phase=["bidon1", "bidon2"]):

    """
    interactive selection of phase among the lit list-phase
    
    Arguments:
        list_phase list: list of phase used for the selection
        
    Returns:
        list of selected phase without duplicate
    """

    # Standard Library dependencies
    import tkinter as tk
    from tkinter import ttk

    global idx
    root = tk.Tk()
    root.title("phases selections.")
    list_phase = list_phase + ["no phase"]

    phase = []
    idx = 2

    def grab_and_assign(event):
        global idx
        chosen_option = myCombo.get()
        label_chosen_variable = tk.Label(root, text=chosen_option)
        label_chosen_variable.grid(row=1, column=idx)
        idx += 1
        phase.append(chosen_option)

    myCombo = ttk.Combobox(root, value=list_phase)
    myCombo.grid(row=0, column=0)
    myCombo.current(0)
    myCombo.bind("<<ComboboxSelected>>", grab_and_assign)

    label_left = tk.Label(root, text="chosen phases = ")
    label_left.grid(row=1, column=0)

    root.mainloop()

    return list(set(phase))
