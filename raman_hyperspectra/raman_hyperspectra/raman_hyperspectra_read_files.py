"""
Read toolbox of Raman spectrum/hyperspectrm with .txt format. The files are issued from:
    - WITEK or;
    - RENISHAW

The spectrum or hyperspectrum is packed in four xarrays:
           - da_sliced : spectum sliced with lbd_dep< lbd < lbd_end with a non uniform sampling interval
           - da_sliced_interp : spectum sliced with lbd_dep< lbd < lbd_end with a uniform sampling interval (fequal to the minimum smpling interval of da)
           - da :  full spectum
           - da_interp : full spectum with a uniform sampling interval (fixed to the minimum smpling interval of da)
The structure of the xarray is as follow:
<xarray.DataArray 'Intensity' (y: 101, x: 101, lbd: 244)>
array([[[...]]])
Coordinates:
  * y        (y) float64 0.0 0.5 1.0 1.5 2.0 2.5 ... 48.0 48.5 49.0 49.5 50.0
  * x        (x) float64 0.0 0.5 1.0 1.5 2.0 2.5 ... 48.0 48.5 49.0 49.5 50.0
  * lbd      (lbd) float64 101.8 106.6 111.4 ... 1.186e+03 1.191e+03 1.195e+03
Attributes:
    units:              u.a.
    tool:               WITEK
    hyperspectra_name:  Large Area Scan_000_Spec_As cut.txt

lbd.attrs = {'units': 'cm-1',
 'spectral_resolution': 4.497769547325103,
 'shift correction': 'yes',
 'spectrum type': 'sliced raw'}

3rd parties dependencies : numpy, pandas, xarray
"""

__all__ = [
    "read_RAMAN_WITEC_0D",
    "read_RAMAN_WITEC_2D",
    "read_RAMAN_WITEC_information",
    "read_RAMAN_RENISHAW_txt_0D",
    "read_RAMAN_RENISHAW_txt_2D",
    "construct_xarray",
    "construct_xarray_0D",
    "read_RENISHAW_1D_wdf",
    "read_RENISHAW_2D_wdf",
]


def read_RAMAN_WITEC_0D(file, lbd_dep=None, lbd_end=None):

    """
    Read a Raman spectra WITEC stored in .txt format

    Arguments:
      file (string): name of the file containing the hyperspectra
      lbd_dep (real): first wavelength [cm-1] (default = None)
      led_end (real): end wavelength [cm-1] (default = None)

    Returns:
      da (xarray): full spectrum
      da_interp (xarray): full spectrum interpollated on a regular grid
      da_sliced (xarray): sliced spectrum between ldb_dep and lbd_end
      da_sliced_interp (xarray): sliced spectrum between ldb_dep and lbd_end interpollated on a constant grid

    """

    # Standard Library dependencies
    import os

    # 3rd party dependencies
    import numpy as np
    import pandas as pd

    data = pd.read_csv(file, sep="\t", engine="python")
    data.columns = ["lbd", "intensity"]

    lbd = np.array(data["lbd"])

    if not lbd_dep:
        lbd_dep = lbd[0]
    if not lbd_end:
        lbd_end = lbd[-1]

    assert lbd_end > lbd_dep, "lbd_dep must be lower than lbd_end"

    spectrum = np.array(data["intensity"])
    hyperspectra_name = os.path.basename(file)

    da = construct_xarray_0D(
        spectrum, lbd, tool="WITEK", hyperspectra_name=hyperspectra_name
    )

    lbd = np.arange(da.lbd.values[0], da.lbd.values[-1], min(np.diff(da.lbd.values)))
    spectrum = da.interp(lbd=lbd, method="cubic")
    da_interp = construct_xarray_0D(
        spectrum, lbd, tool="WITEK", hyperspectra_name=hyperspectra_name
    )

    da_sliced = da.sel(lbd=slice(lbd_dep, lbd_end))

    lbd = np.arange(
        da_sliced.lbd.values[0],
        da_sliced.lbd.values[-1],
        min(np.diff(da_sliced.lbd.values)),
    )
    spectrum = da_sliced.interp(lbd=lbd, method="cubic")
    da_sliced_interp = construct_xarray_0D(
        spectrum, lbd, tool="WITEK", hyperspectra_name=hyperspectra_name
    )

    return da_sliced, da_sliced_interp, da, da_interp


def construct_xarray_0D(spectrum, lbd, tool="WITEK", hyperspectra_name="unknown"):

    """
    construct the xarray of a spectrum

    Arguments:
        spectrum (np array): spectrum intensities
        lbd (np array): vawelengths
        tool (string) : name of thRaman equipment
        hyperspectra_nane (sring): name of the hyperspectra

    Returns:
        da (xarray): spectrum
    """

    # 3rd party dependencies
    import numpy as np
    import xarray as xr

    spectral_resolution = np.mean(np.diff(lbd))

    da = xr.DataArray(
        spectrum,
        dims=["lbd"],
        name="Intensity",
        attrs={"units": "u.a.", "tool": tool, "hyperspectra_name": hyperspectra_name},
        coords={
            "lbd": xr.DataArray(
                lbd,
                name="lbd",
                dims=["lbd"],
                attrs={
                    "units": "cm-1",
                    "spectral_resolution": spectral_resolution,
                    "shift correction": "no",
                },
            )
        },
    )
    return da


def construct_xarray(
    Intensity, x, y, lbd, units="u.a.", tool="WITEK", hyperspectra_name="unknown"
):

    """
    construct the xarray of a hyperspectrum

    Arguments:
        spectrum (np array): hyperspectrum intensities values
        lbd (np array): vawelengths
        x (np array): x location of the spectrum express in index (WITEk), distance (RENISHAW)
        y (np array): y location of the spectrum express in index (WITEk), distance (RENISHAW)
        units (string): 'u.a.' for WITEK, 'µm' for RENISHAW
        tools (string): 'WITEK' or 'RENISHAW'
        hyperspectra_nane (sring): name of the hyperspectra

    Returns:
        da (xarray): hyperspectrum
    """

    # 3rd party dependencies
    import xarray as xr
    import numpy as np

    spectral_resolution = np.mean(np.diff(lbd))

    da = xr.DataArray(
        Intensity,
        dims=["y", "x", "lbd"],
        name="Intensity",
        attrs={"units": "u.a.", "tool": tool, "hyperspectra_name": hyperspectra_name},
        coords={
            "y": xr.DataArray(x, name="y", dims=["y"], attrs={"units": units}),
            "x": xr.DataArray(y, name="x", dims=["x"], attrs={"units": units}),
            "lbd": xr.DataArray(
                lbd,
                name="lbd",
                dims=["lbd"],
                attrs={
                    "units": "cm-1",
                    "spectral_resolution": spectral_resolution,
                    "shift correction": "no",
                },
            ),
        },
    )

    return da


def read_RAMAN_WITEC_2D(file, lbd_dep=None, lbd_end=None):

    """
    Read a WITEC Raman hyperspectrastored in .txt format
    The first  column label is : 'X-Axis'
        second                 : 'Large Area Scan_000_Spec.Data 1(0/0)'
        last                   : 'Large Area Scan_000_Spec.Data 1(I_col/I_row)'
    where I_col and Irow are the latest index of the column and of the row.

    The first column contains the vawelength array lbd(I) i=0,...,N_row-1.
    The other columns contain the N_col*N_row spectra. These spectra are packed in
    a two-dimentional z(K,L) where K is related to the pixel coordinates I,J K=J*N_row+J.
    L is the index of the vawelength with lbd_dep<=lbd<=lbd_end.

    Arguments:
      file (string): name of the file containing the hyperspectra
      lbd_dep (real): first wavelength [cm-1] (default = None)
      led_end (real): end wavelength [cm-1] (default = None)

    Returns:
      da (xarray): full spectrum
      da_interp (xarray): full spectrum interpollated on a regular grid
      da_sliced (xarray): sliced spectrum between ldb_dep and lbd_end
      da_sliced_interp (xarray): sliced spectrum between ldb_dep and lbd_end interpollated on a constant grid

    """

    # Standard Library dependencies
    import re
    import os

    # 3rd party dependencies
    import pandas as pd
    import numpy as np

    data = pd.read_csv(file, sep="\t", engine="python")

    motif = re.compile("\d+")
    N_col, N_row = re.findall(motif, data.columns[-1])[
        -2:
    ]  # find all numbers and select the two last
    N_col = int(N_col) + 1
    N_row = int(N_row) + 1

    lbd = np.array(
        data[data.columns[0]]
    )  # in cm-1 the sampling interval is not constant

    if not lbd_dep:
        lbd_dep = lbd[0]
    if not lbd_end:
        lbd_end = lbd[-1]

    assert lbd_end > lbd_dep, "lbd_dep must be lower than lbd_end"

    spectrums = data.values.T[1 : N_col * N_row + 1, :]
    spectrums = spectrums.reshape((N_row, N_col, -1))
    spectrums = spectrums[::-1, :, :]

    hyperspectra_name = os.path.basename(file)

    da = construct_xarray(
        spectrums,
        range(N_row),
        range(N_col),
        lbd,
        tool="WITEK",
        hyperspectra_name=hyperspectra_name,
    )
    da.lbd.attrs["spectrum type"] = "full raw"

    lbd_interp = np.arange(
        da.lbd.values[0], da.lbd.values[-1], min(np.diff(da.lbd.values))
    )
    da_interp = da.interp(lbd=lbd_interp, method="cubic")
    da_interp.lbd.attrs = {
        "spectral_resolution": min(np.diff(da.lbd.values)),
        "units": "cm-1",
        "spectrum type": "full interp",
        "shift correction": "no",
    }

    da_sliced = da.sel(lbd=slice(lbd_dep, lbd_end))
    da_sliced.lbd.attrs = {
        "spectral_resolution": min(np.diff(da.lbd.values)),
        "units": "cm-1",
        "spectrum type": "sliced raw",
        "shift correction": "no",
    }

    lbd_interp = np.arange(
        da_sliced.lbd.values[0],
        da_sliced.lbd.values[-1],
        min(np.diff(da_sliced.lbd.values)),
    )
    da_sliced_interp = da_sliced.interp(lbd=lbd_interp, method="cubic")
    da_sliced_interp.lbd.attrs = {
        "spectral_resolution": np.mean(np.diff(da_sliced_interp.lbd.values)),
        "units": "cm-1",
        "spectrum type": "sliced interp",
        "shift correction": "no",
    }

    return da_sliced, da_sliced_interp, da, da_interp


def read_RAMAN_WITEC_information(file):

    """
    Read an information file, file, provided by WITEC.

    Arguments:
     file : the documentation file

    Returns:
     dic_info : dictionary with keys : Points per Line, Lines per Image, Scan Width [µm],
                                       Scan Height [µm], Scan Origin X [µm], Scan Origin Y [µm]
                                       Scan Origin Z [µm], Gamma [°], Scan Speed [s/Line],
                                       Integration Time [s]
    """

    # Standard Library dependencies
    import re

    motif = re.compile("-?\d+\.?\d+")
    with open(file) as f:
        content = f.readlines()

    dic_info = {}
    for x in content[11:19]:
        dic_info[x.split(":")[0]] = float(re.findall(motif, x)[-1])

    return dic_info


def read_RAMAN_RENISHAW_txt_0D(file, lbd_dep=None, lbd_end=None):

    """
    Read a Raman spectra RENISHAW stored in .txt format

    Arguments:
      file (string): name of the file containing the hyperspectra
      lbd_dep (real): first wavelength [cm-1] (default = None)
      led_end (real): end wavelength [cm-1] (default = None)

    Returns:
      da (xarray): full spectrum
      da_interp (xarray): full spectrum interpollated on a regular grid
      da_sliced (xarray): sliced spectrum between ldb_dep and lbd_end
      da_sliced_interp (xarray): sliced spectrum between ldb_dep and lbd_end interpollated on a constant grid

    """

    # Standard Library dependencies
    import os

    import numpy as np
    import pandas as pd

    data = pd.read_csv(file, sep="\t", engine="python")

    lbd = np.array(data["#Wave"][::-1])

    if not lbd_dep:
        lbd_dep = lbd[0]
    if not lbd_end:
        lbd_end = lbd[-1]

    assert lbd_end > lbd_dep, "lbd_dep must be lower than lbd_end"

    spectrum = np.array(data["Unnamed: 1"][::-1])  # beware the extra \t

    hyperspectra_name = os.path.basename(file)

    da = construct_xarray_0D(
        spectrum, lbd, tool="RENIHAW", hyperspectra_name=hyperspectra_name
    )

    lbd = np.arange(da.lbd.values[0], da.lbd.values[-1], min(np.diff(da.lbd.values)))
    spectrum = da.interp(lbd=lbd, method="cubic")
    da_interp = construct_xarray_0D(spectrum, lbd, tool="RENIHAW")

    da_sliced = da.sel(lbd=slice(lbd_dep, lbd_end))

    lbd = np.arange(
        da_sliced.lbd.values[0],
        da_sliced.lbd.values[-1],
        min(np.diff(da_sliced.lbd.values)),
    )
    spectrum = da_sliced.interp(lbd=lbd, method="cubic")
    da_sliced_interp = construct_xarray_0D(spectrum, lbd, tool="RENIHAW")

    return da_sliced, da_sliced_interp, da, da_interp


def read_RAMAN_RENISHAW_txt_2D(file, lbd_dep=None, lbd_end=None):
    """
    Read a RENISHAW Raman hyperspectrastored in .txt format

    Arguments:
      file (string): name of the file containing the hyperspectra
      lbd_dep (real): first wavelength [cm-1] (default = None)
      led_end (real): end wavelength [cm-1] (default = None)

    Returns:
      da (xarray): full spectrum
      da_interp (xarray): full spectrum interpollated on a regular grid
      da_sliced (xarray): sliced spectrum between ldb_dep and lbd_end
      da_sliced_interp (xarray): sliced spectrum between ldb_dep and lbd_end interpollated on a constant grid

    """

    # Standard Library dependencies
    import os

    # 3rd party dependencies
    import pandas as pd
    import numpy as np

    data = pd.read_csv(file, sep="\t", engine="python")

    x = sorted(list(set(data["#X"])))
    y = sorted(list(set(data["Unnamed: 1"])))  # beware an extra \t
    N_col = len(x)
    N_row = len(y)

    lbd = np.array(data["#Y"])  # beware an an extra \t
    N_lbd = int(len(lbd) / (N_row * N_col))
    lbd = lbd[0:N_lbd][::-1]

    if not lbd_dep:
        lbd_dep = lbd[0]
    if not lbd_end:
        lbd_end = lbd[-1]

    assert lbd_end > lbd_dep, "lbd_dep must be lower than lbd_end"

    spectrums = np.array(data["Unnamed: 3"])  # beware an extra \t
    spectrums = spectrums.reshape((N_row, N_col, -1))
    # spectrums = spectrums[::-1,:,:]  # flip the image around the x axis
    hyperspectra_name = os.path.basename(file)

    da = construct_xarray(
        spectrums,
        y,
        x,
        lbd,
        units="µm",
        tool="RENISHAW",
        hyperspectra_name=hyperspectra_name,
    )
    da.lbd.attrs["spectrum type"] = "full raw"

    lbd_interp = np.arange(
        da.lbd.values[0], da.lbd.values[-1], min(np.diff(da.lbd.values))
    )
    da_interp = da.interp(lbd=lbd_interp, method="cubic")
    da_interp.lbd.attrs = {
        "spectral_resolution": min(np.diff(da.lbd.values)),
        "units": "cm-1",
        "spectrum type": "full interp",
        "shift correction": "no",
    }

    da_sliced = da.sel(lbd=slice(lbd_dep, lbd_end))
    da_sliced.lbd.attrs = {
        "spectral_resolution": min(np.diff(da.lbd.values)),
        "units": "cm-1",
        "spectrum type": "sliced raw",
        "shift correction": "no",
    }

    lbd_interp = np.arange(
        da_sliced.lbd.values[0],
        da_sliced.lbd.values[-1],
        min(np.diff(da_sliced.lbd.values)),
    )
    da_sliced_interp = da_sliced.interp(lbd=lbd_interp, method="cubic")
    da_sliced_interp.lbd.attrs["spectral_resolution"] = np.mean(
        np.diff(da_sliced_interp.lbd.values)
    )
    da_sliced_interp.lbd.attrs["units"] = "cm-1"
    da_sliced_interp.lbd.attrs["spectrum type"] = "sliced interpolated"

    return da_sliced, da_sliced_interp, da, da_interp


def read_RENISHAW_1D_wdf(file, lbd_dep=None, lbd_end=None):

    """
    Read a Raman spectra stored in .wdf (RENIHAW format)
    base on the package renishawWiRE
    We assert that the spectral unit is expressed in cm-1 (Raman shift)

    Arguments:
      file (string): name of the file containing the hyperspectra
      lbd_dep (real): first wavelength [cm-1] (default = None)
      led_end (real): end wavelength [cm-1] (default = None)

    Returns:
      da (xarray): full spectrum
      da_interp (xarray): full spectrum interpollated on a regular grid
      da_sliced (xarray): sliced spectrum between ldb_dep and lbd_end
      da_sliced_interp (xarray): sliced spectrum between ldb_dep and lbd_end interpollated on a constant grid

    """

    # Standard Library dependencies
    import os

    # 3rd party dependencies
    import numpy as np
    import pandas as pd
    import renishawWiRE as wire
    from renishawWiRE import types

    if not file.lower().endswith(".wdf"):
        raise Exception("not a .wdf file")

    data = wire.WDFReader(file)

    assert (
        data.xlist_unit == 1
    ), "error in read_RENISHAW_1D_wdf : not a Raman shift measurement"
    lbd = data.xdata[::-1]

    if not lbd_dep:
        lbd_dep = lbd[0]
    if not lbd_end:
        lbd_end = lbd[-1]

    assert lbd_end > lbd_dep, "lbd_dep must be lower than lbd_end"

    spectrum = data.spectra

    hyperspectra_name = os.path.basename(file)

    da = construct_xarray_0D(
        spectrum, lbd, tool="RENIHAW", hyperspectra_name=hyperspectra_name
    )

    lbd = np.arange(da.lbd.values[0], da.lbd.values[-1], min(np.diff(da.lbd.values)))
    spectrum = da.interp(lbd=lbd, method="cubic")
    da_interp = construct_xarray_0D(spectrum, lbd, tool="RENIHAW")

    da_sliced = da.sel(lbd=slice(lbd_dep, lbd_end))

    lbd = np.arange(
        da_sliced.lbd.values[0],
        da_sliced.lbd.values[-1],
        min(np.diff(da_sliced.lbd.values)),
    )
    spectrum = da_sliced.interp(lbd=lbd, method="cubic")
    da_sliced_interp = construct_xarray_0D(spectrum, lbd, tool="RENIHAW")

    return da_sliced, da_sliced_interp, da, da_interp


def read_RENISHAW_2D_wdf(file, lbd_dep=None, lbd_end=None):

    """
    Read a Raman hyperspectra stored in .wdf (RENISHAW format)
    base on the package renishawWiRE.
    We assert that: (i)the spectral unit is expressed in cm-1 (Raman shift); (ii) the unit of x and y axes are equal

    Arguments:
      file (string): name of the file containing the hyperspectra
      lbd_dep (real): first wavelength [cm-1] (default = None)
      led_end (real): end wavelength [cm-1] (default = None)

    Returns:
      da (xarray): full spectrum
      da_interp (xarray): full spectrum interpollated on a regular grid
      da_sliced (xarray): sliced spectrum between ldb_dep and lbd_end
      da_sliced_interp (xarray): sliced spectrum between ldb_dep and lbd_end interpollated on a constant grid

    """

    # Standard Library dependencies
    import os

    # 3rd party dependencies
    import numpy as np
    import pandas as pd
    import renishawWiRE as wire
    from renishawWiRE import types

    if not file.lower().endswith(".wdf"):
        raise Exception("not a .wdf file")

    data = wire.WDFReader(file, quiet=True)

    assert data.measurement_type == 3, "error in read_RENISHAW_2D_wdf : not a map"
    assert (
        data.xlist_units == 1
    ), "error in read_RENISHAW_2D_wdf : not a Raman shift measurement"

    lbd = data.xdata[::-1]

    if not lbd_dep:
        lbd_dep = lbd[0]
    if not lbd_end:
        lbd_end = lbd[-1]

    assert lbd_end > lbd_dep, "lbd_dep must be lower than lbd_end"

    unit_x = data.origin_list_header[0][2].name
    if unit_x == "Micron":
        unit_x = "µm"
    x = np.unique(data.xpos)
    x -= x[0]

    unit_y = data.origin_list_header[1][
        2
    ].name  # for future use in the following we assert unit_x=unit_y
    if unit_y == "Micron":
        unit_y = "µm"
    y = np.unique(data.ypos)
    y -= y[0]
    spectrum = data.spectra

    hyperspectra_name = os.path.basename(file)

    da = construct_xarray(
        spectrum,
        y,
        x,
        lbd,
        units=unit_x,
        tool="RENISHAW",
        hyperspectra_name=hyperspectra_name,
    )
    da.lbd.attrs["spectrum type"] = "full raw"

    lbd_interp = np.arange(
        da.lbd.values[0], da.lbd.values[-1], min(np.diff(da.lbd.values))
    )
    da_interp = da.interp(lbd=lbd_interp, method="cubic")
    da_interp.lbd.attrs["spectral_resolution"] = min(np.diff(da.lbd.values))
    da_interp.lbd.attrs["units"] = "cm-1"
    da_interp.lbd.attrs["spectrum type"] = "full interpolated"

    da_sliced = da.sel(lbd=slice(lbd_dep, lbd_end))
    da_sliced.lbd.attrs["spectral_resolution"] = np.mean(np.diff(da_sliced.lbd.values))
    da_sliced.lbd.attrs["spectrum type"] = "sliced raw"

    lbd_interp = np.arange(
        da_sliced.lbd.values[0],
        da_sliced.lbd.values[-1],
        min(np.diff(da_sliced.lbd.values)),
    )
    da_sliced_interp = da_sliced.interp(lbd=lbd_interp, method="cubic")
    da_sliced_interp.lbd.attrs = {
        "spectral_resolution": np.mean(np.diff(da_sliced_interp.lbd.values)),
        "units": "cm-1",
        "spectrum type": "sliced interp",
        "shift correction": "no",
    }

    return da_sliced, da_sliced_interp, da, da_interp
