"""
Fits, using  the mean square method, a spectra/hyperstra. The ojective function is a sum of:
    - Gaussian(s)
    - bigaussian(s)
    - Lorentzian(s)
    - Voigt(s)
    - pseudo Voigt(s)
    
The fitting parameters are stored in an Excel/csv file. 

To obtain an initialization file, for a Gaussian model execute the following script:
with open(r'c:\Temp\modele.csv','w') as file:
    file.write("FITTING MODEL INITIALIZATION\n")
    file.write("Function type (column A) : Gaussian, Lorentzian, Bigaussian, Voigt, Pseudo_Voigt\n")
    file.write("B6 offset, B7,B8... C7,C8...must be blank. mask=1/0 the parameter is free/freezed.\
               Lbd_0, sigma_a, gamma are expressed in cm^-1\n")
    file.write(" ;offset;mask;Lbd_0;mask;h;mask;sigma_a (w_a);mask;sigma_b;mask;gamma;mask;gamma;mask\n\n")
    file.write("Gaussian;0;0;521;1;5000;1;5;1;;;;;;")

This file is parsed by the function init_model:

     param_fixed ,param_fit, func_type, index_fit, label = rhp.init_model(r'c:\Temp\modele.csv')
     print(f'param_fixed :{param_fixed}',f'param_fit :{param_fit}',f'func_type :{list(func_type)}' , \
      f'index_fit: {index_fit}',f'label: {label} ', sep='\\n')
      
    >>>param_fixed :[0.0, 0, 0, 0]
    >>>param_fit :[521.0, 10.0, 5.0]
    >>>func_type :['Gaussian0']
    >>>index_fit: [1, 2, 3]
    >>>label: ['Gaussian0  :  Lbd_0 = 521.0 ', 'Gaussian0  :  h = 10.0 ', 'Gaussian0  :  sigma_a = 5.0 '] 



For a Pseudo_Voigt model execute the following script:
with open(r'c:\Temp\modele.csv','w') as file:
    file.write("FITTING MODEL INITIALIZATION\n")
    file.write("Function type (column A) : Gaussian, Lorentzian, Bigaussian, Voigt, Pseudo_Voigt\n")
    file.write("B6 offset, B7,B8... C7,C8...must be blank. mask=1/0 the parameter is free/freezed.\
               Lbd_0, sigma_a, gamma are expressed in cm^-1\n")
    file.write(" ;offset;mask;Lbd_0;mask;h;mask;sigma_a (w_a);mask;sigma_b;mask;gamma;mask;gamma;mask\n\n")
    file.write("Pseudo_Voigt;0;0;521;1;28000;1;10;1;;;101;0.3;1;1")

For a Lorentzian model, execute the following script:
with open(r'c:\Temp\modele.csv','w') as file:
    file.write("FITTING MODEL INITIALIZATION\n")
    file.write("Function type (column A) : Gaussian, Lorentzian, Bigaussian, Voigt, Pseudo_Voigt\n")
    file.write("B6 offset, B7,B8... C7,C8...must be blank. mask=1/0 the parameter is free/freezed.\
               Lbd_0, sigma_a, gamma are expressed in cm^-1\n")
    file.write(" ;offset;mask;Lbd_0;mask;h;mask;sigma_a (w_a);mask;sigma_b;mask;gamma;mask;gamma;mask\n\n")
    file.write("Lorentzian;0;1;520;1;10000;1;;;;;5;1")		

For a Voigt model execute th following script:
with open(r'c:\Temp\modele.csv','w') as file:
    file.write("FITTING MODEL INITIALIZATION\n")
    file.write("Function type (column A) : Gaussian, Lorentzian, Bigaussian, Voigt, Pseudo_Voigt\n")
    file.write("B6 offset, B7,B8... C7,C8...must be blank. mask=1/0 the parameter is free/freezed.\
               Lbd_0, sigma_a, gamma are expressed in cm^-1\n")
    file.write(" ;offset;mask;Lbd_0;mask;h;mask;sigma_a (w_a);mask;sigma_b;mask;gamma;mask;gamma;mask\n\n")
    file.write("Voigt;0;1;520;1;28000;1;5;1;;;1;1;;")

for a sum of a Gauusian and a Lorentzian , execute the following script:
with open(r'c:\Temp\modele.csv','w') as file:
    file.write("FITTING MODEL INITIALIZATION\n")
    file.write("Function type (column A) : Gaussian, Lorentzian, Bigaussian, Voigt, Pseudo_Voigt\n")
    file.write("B6 offset, B7,B8... C7,C8...must be blank. mask=1/0 the parameter is free/freezed.\
               Lbd_0, sigma_a, gamma are expressed in cm^-1\n")
    file.write(" ;offset;mask;Lbd_0;mask;h;mask;sigma_a (w_a);mask;sigma_b;mask;gamma;mask;gamma;mask\n\n")
    file.write("Gaussian;0;0;521;1;5000;1;5;1;;;;;;\n")
    file.write("Lorentzian;;;400;1;1000;0;;;;;5;1")



Internal functions:
 wrapper_fit_func

3rd party dependencies: pandas, numpy, scipy, sklearn, xarray
"""


__all__ = ["fit_Raman", "fit_hyperspectra_Si", "init_model", "sum_functions"]


def init_model(file, sheet=None):

    """
    init_model parses the Excel/cv file containing the fiting model
    for more information execute:
        import raman_hyperspectra as rhp
        print(rhp.raman_hyperspectra_fit.__doc__)
    Follow the instruction to create your own init_file

   Arguments:
        file (string): path+name of the Excel/csv storing the modele
        sheet (string): name of the sheet (default = None)

   Returns:
        param_fixed (ndarray): list of the values of both the freezed and the free parameters. The free parameters are set to 0.
        param_fit (ndarray): list of the values of the free parameters
        type (list): list of fonctions to sum
        mask (ndarray): list of parameter mask (1 is the parameter is fee, 0 if the parameter is freezed)
        index_fit (ndarray): list of the indice of the free parameters in the ndarray 'param_fixed'
        label (list): ['function type : name of the free parater',... ])

    """

    # 3rd party import
    import pandas as pd
    import numpy as np

    mask_param = {
        "mask": "Offset",
        "mask.1": "Lbd_0",
        "mask.2": "h",
        "mask.3": "sigma_a",
        "mask.4": "sigma_b",
        "mask.5": "gamma",
        "mask.6": "eta",
    }
    label = []

    if file.lower().endswith(".csv"):
        data = pd.read_csv(file, sep=";", header=3, index_col=0)

    elif file.lower().endswith(".xlsx") or file.lower().endswith(".xls"):
        if sheet is not None:
            data = pd.read_excel(file, header=4, index_col=0, sheet_name=sheet)
        else:
            data = pd.read_excel(file, header=4, index_col=0)
    else:
        raise Exception(
            "init_param: file extension not recognize sould be csv, xls or xlsx"
        )

    data.index = data.index + data.groupby(level=0).cumcount().astype(
        str
    )  # increment repeated index (ex. Gauss,Gauss ->Gauss0,Gauss1)
    func_type = data.index

    param = data.values.flatten()
    param = param[np.logical_not(np.isnan(param))]
    param_0 = param[0::2]
    mask = param[1::2]
    param_fit = np.array([param_0[i] for i in range(len(param_0)) if mask[i]])
    index_fit = np.array([i for i in range(len(param_0)) if mask[i]])
    param_fixed = np.array([x[0] if x[1] == 0 else 0 for x in zip(param_0, mask)])

    for row in data.iterrows():
        mask_fit = row[1][1::2]  # every two columns is a mask
        mask_fit = mask_fit[mask_fit == 1]
        for x in list(mask_fit.index):
            try:
                idx = int(x.split(".")[1])
            except:
                idx = 0

            label = [*label, f"{row[0]}  :  {mask_param[x]} = {row[1] [2*idx] } "]

    return (param_fixed, param_fit, func_type, index_fit, label)


def sum_functions(lbd, p0, func_types, index_fit, param_fixed):

    """
    Sum of the funtions which define the model. The parameter h is the area of the function
    not the height of the peak.
    
    Arguments:
        lbd (ndarray): wavelengths
        p0 (ndrray): parameters to be fitted
        func_type (list): list of function to be added
        index_fit (ndarray): list of the indice of the free parameters in the ndarray 'param_fixed'
        param_fixed (ndarray): array of the fixed parameter
      
    Returns:
      sg (ndarray): computed intensities

    """

    # 3rd party import
    from scipy.special import wofz
    import numpy as np

    param = param_fixed  # merge free and freezed parameters in one array
    if len(index_fit) != 0:
        param[index_fit] = p0

    idx = 0
    sg = param[0]  # initialize the sum with the offset
    for func_type in func_types:

        if func_type.lower().strip().startswith("gaussian"):
            lbd0, h, sigma = param[idx + 1 : idx + 4]
            sg += (
                h
                * np.exp(-((lbd - lbd0) ** 2) / (2 * sigma ** 2))
                / (np.sqrt(2 * np.pi) * sigma)
            )
            idx += 3

        elif func_type.lower().strip().startswith("lorentzian"):
            lbd0, h, gamma = param[idx + 1 : idx + 4]
            sg += h * gamma / 2 * 1 / (np.pi * ((lbd - lbd0) ** 2 + (gamma / 2) ** 2))
            idx += 3

        elif func_type.lower().strip().startswith("bigaussian"):
            lbd0, h, sigma_a, sigma_b = param[idx + 1 : idx + 5]
            fact = 2 / (np.sqrt(2 * np.pi) * (sigma_a + sigma_b))
            sg += (
                h
                * fact
                * (1 - np.heaviside(lbd - lbd0, 1))
                * np.exp(-(((lbd - lbd0) / sigma_a) ** 2) / 2)
            )
            sg += (
                h
                * fact
                * np.heaviside(lbd - lbd0, 1)
                * np.exp(-(((lbd - lbd0) / sigma_b) ** 2) / 2)
            )
            idx += 4

        elif func_type.lower().strip().startswith("voigt"):
            e, h, alpha, gamma = param[idx + 1 : idx + 5]
            sigma = alpha / np.sqrt(2 * np.log(2))
            sg += (
                h
                * np.real(wofz((lbd - e + 1j * gamma) / sigma / np.sqrt(2)))
                / sigma
                / np.sqrt(2 * np.pi)
            )
            idx += 4

        elif func_type.lower().strip().startswith("pseudo_voigt"):
            e, h, alpha, gamma, eta = param[idx + 1 : idx + 6]
            sigma = alpha / np.sqrt(8 * np.log(2))  # corrigé le 01/03/2018
            sg += h * (
                eta * gamma / 2 * 1 / (np.pi * ((lbd - e) ** 2 + (gamma / 2) ** 2))
                + (1 - eta) * np.exp(-((lbd - e) ** 2) / (2 * sigma ** 2))
                / (np.sqrt(2 * np.pi) * sigma)
            )
            idx += 5

        else:
            raise ValueError(
                f"sum_functions : unknown function {func_type}  the function name must begin with: Gaussian or Lorentzian or Bigaussian \
                                or Voigt or Pseudo_Voigt"
            )

    return sg


def wrapper_fit_func(x, func_type, index_fit, param_fixed, param_fit):

    """
    wrapper function for curve_fit(f, xdata, ydata, p0 = param_fit) 
    where  f(x, *param_fit) is the objective function
    
    Arguments:
        x(ndarray): vawelengths
        func_type (list): list of function to be added
        index_fit (ndarray): list of the indice of the free parameters in the ndarray 'param_fixed'
        param_fixed (ndarray): array of the fixed parameter
        param_fit (ndrray): parameters to be fitted
      
    Returns:
        y_model (ndarray): value of the fiiting function with parameter param_fit
        
    """
    y_model = sum_functions(x, param_fit, func_type, index_fit, param_fixed)
    return y_model


def fit_Raman(da, file_model, sheet=""):

    """
    fits with least square method a spectrum. The parameters of the objective function
    are defined in xlsx or in a csv file. 
    For an xlsx file the name of the sheet can be specified (optional)
    
    To create your own mode execute:
        import raman_hyperspectra as rhp
        print(rhp.raman_hyperspectra_fit.__doc__)
        follow the instructions
    
    Arguments:
        da (DataArray): xarray containing the spectrum
        fit_model (string): the absolute path of the Excel file containing the model parameters
        sheet (string): name of the Excel file sheet containing the model parameter (default = None)
        
    Returns:
        popt (float): fitting parameters
        da_fit (DataArray): fitted spectrum
        R2 (float): fit determination coefficient
    """

    # Standard library import
    import warnings

    # 3rd party import
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score
    import numpy as np
    import xarray as xr

    warnings.filterwarnings("ignore")  # ignore scipy warnings

    # Error handling
    if not type(da).__name__ == "DataArray":
        raise TypeError("fit_Raman: 'da' must be an DataArray.")

    lbd = da["lbd"].data
    y_exp = da.data

    fit_Raman.flag = getattr(fit_Raman, "flag", True)

    if fit_Raman.flag:  # First call
        param_fixed, param_fit, func_type, index_fit, label = init_model(
            file_model, sheet
        )

        fit_Raman.param_fixed = getattr(fit_Raman, "param_fixed", param_fixed)
        fit_Raman.param_fit = getattr(fit_Raman, "param_fit", param_fit)
        fit_Raman.func_type = getattr(fit_Raman, "func_type", func_type)
        fit_Raman.index_fit = getattr(fit_Raman, "index_fit", index_fit)
        fit_Raman.label = getattr(fit_Raman, "label", label)

        fit_Raman.flag = False
        fit_Raman.model = getattr(fit_Raman, "model", file_model + sheet)

    elif fit_Raman.model != file_model + sheet:  # model change
        param_fixed, param_fit, func_type, index_fit, label = init_model(
            file_model, sheet
        )

        fit_Raman.param_fixed = param_fixed
        fit_Raman.param_fit = param_fit
        fit_Raman.func_type = func_type
        fit_Raman.index_fit = index_fit
        fit_Raman.label = label

        fit_Raman.model = file_model + sheet

    else:
        param_fixed = fit_Raman.param_fixed
        param_fit = fit_Raman.param_fit
        func_type = fit_Raman.func_type
        index_fit = fit_Raman.index_fit
        label = fit_Raman.label

    try:

        popt, _ = curve_fit(
            lambda lbd, *param_fit: wrapper_fit_func(
                lbd, func_type, index_fit, param_fixed, param_fit
            ),
            lbd,
            y_exp,
            p0=param_fit,
        )

        y_fit = sum_functions(lbd, popt, func_type, index_fit, param_fixed)
        R2 = r2_score(y_exp, y_fit)  # computes the coefficient of determination

        da_fit = xr.DataArray(
            y_fit,
            dims=["lbd"],
            coords={
                "lbd": xr.DataArray(
                    lbd, name="lbd", dims=["lbd"], attrs={"units": "cm-1"}
                )
            },
        )
    except:

        popt = [np.nan] * len(param_fit)
        R2 = np.nan
        da_fit = None

    return popt, da_fit, R2


def fit_hyperspectra_Si(
    da, file_model, sheet="", lbd_deb_fit=None, lbd_end_fit=None, save_xlsx=True
):

    """
    This function fits an hyperspectra using the modele store in the csv/xlsx file "file_model"
    
    To create your own mode execute:
        import raman_hyperspectra as rhp
        print(rhp.raman_hyperspectra_fit.__doc__)
        follow the instructions
    
    Arguments:
      da (DataArray): hyperstrum
      file_model (string): the Excel file containing the model
      sheet (string): name of the sheet containing the model (default = '')
      lbd_deb_fit (float): (optional) the first vawelength of the fitting domain
      lbd_end_fit (float): (optional) the last vawelength of the fitting domain
      save_xlsx (bool): if True save the results as 'Si cristallin.xlsx'

    Returns:
      df (multi index dataFrame): fitted parameters and the goodness of fit parameter (r^2)
    """

    # Standard library import
    import warnings

    # 3rd party import
    import pandas as pd
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score
    import numpy as np
    from tqdm import trange

    warnings.filterwarnings("ignore")  # ignore scipy warnings

    # Error handling
    if not type(da).__name__ == "DataArray":
        raise TypeError("fit_hyperspectra_Si: 'da' must be an DataArray.")

    lbd = da.lbd.data

    if lbd_deb_fit is None:
        lbd_deb_fit = lbd[0]

    if lbd_end_fit is None:
        lbd_end_fit = lbd[-1]

    param_fixed, param_fit, func_type, index_fit, label = init_model(file_model, sheet)

    da_fit = da.sel(lbd=slice(lbd_deb_fit, lbd_end_fit))
    lbd = da_fit.lbd.values
    Ncol = da.shape[0]
    Nrow = da.shape[1]
    y = da_fit.values.reshape((Ncol * Nrow, -1))
    dic = {}

    for idx in trange(Ncol * Nrow, desc=f"fit ({func_type[0][0:-1]})"):

        try:

            popt, _ = curve_fit(
                lambda lbd, *param_fit: wrapper_fit_func(
                    lbd, func_type, index_fit, param_fixed, param_fit
                ),
                lbd,
                y[idx],
                p0=param_fit,
            )

            y_fit = sum_functions(lbd, popt, func_type, index_fit, param_fixed)
            R2 = r2_score(y[idx], y_fit)

        except:

            popt = [np.nan] * len(param_fit)
            R2 = np.nan
            da_fit = None

        dic[(idx // Ncol, idx % Ncol)] = list(popt) + [R2]

    df = pd.DataFrame(dic).T
    df.columns = [x.split("=")[0].strip() for x in label] + ["R2"]

    if save_xlsx:
        df.to_excel("Si cristallin.xlsx")

    return df
