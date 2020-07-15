def phase_si() :   
    import raman_hyperspectra as rhp
    import warnings
    import numpy as np
    from functools import reduce
    import matplotlib.pyplot as plt

    warnings.filterwarnings("ignore")

    file_init, user_path = rhp.users_init_path()

    dic_ref, class_Si, dic_bornes,par, \
    file_raw, file_info, path_save, path_image, \
    filename = rhp.init_raman_hyperspectra(file_init, user_path)

    print("file reading...")

    da_raw, da_raw_interp, da_full_spectrum, da_full_spectrum_interp  = rhp.read_RAMAN_WITEC_2D(file_raw,
                                                                                               par.lbd_dep,
                                                                                               par.lbd_end)
    choice_file, choice_method, choice_shift = rhp.mode_selection()
    print(choice_method)

    if choice_file == "spectrum" :
      da = da_raw

    elif choice_file == "interp spectrum" :
      da = da_raw_interp

    elif choice_file == "full spectrum" :
      da = da_full_spectrum

    elif choice_file == "interp full spectrum" :
      da = da_full_spectrum_interp


    par.factor = 440 # [cm-1]
    da_flatten = rhp.flatten_hyperspectra(da_raw,
                                          ("top hat", par.factor))        # ('linear',500,5)
                                                                                # ('als',10_000, 0.001,20)
                                                                                # ("top hat", par.factor)
                                                                                # ('ials',10_000,  0.01, 0.001)
    print("pass flatten the hyperspectrum")

    da_cosmic = rhp.cosmic_cleaning(da_flatten,
                                    Kmax = par.Kmax)

    print("pass clean the hyperspectrum from cosmics")

    da_noise = rhp.spectra_noise_estimation(da_cosmic,
                                           ("Savitzky-Golay",7, 3),
                                           ("Gauss",1.7),
                                           lbd_inf_var = 100,
                                           lbd_sup_var = 480)

    print("pass noise evaluation of the hyperspectrum ")

    da_denoise = rhp.denoise_hyperspectra(da_cosmic, ("Savitzky-Golay",7, 3))
                                           #('NRWRT',
                                           #da_noise.sel(Filter = 'Gauss').values,
                                           #par.decNum,
                                           #par.wname,
                                           #par.mode,
                                           #0.5))

    print("pass denoise the hyperspectrum ")

    da_std = rhp.noise_plot( da_denoise,
                             da_cosmic,
                             da_noise,
                             bins =par.bins_noise,
                             style = par.style_noise,
                             save = True,
                             robust = True )

    if choice_shift == "Raman peak" :
        d_lbd_Si = rhp.set_peak_to_value(da_denoise,
                                    par.lbd_si_c_1,
                                    file_init,
                                    sheet = par.file_model,
                                    tol_R2 = par.R_cSi,
                                    lbd_deb_fit = par.lbd_deb_fit,
                                    lbd_end_fit = par.lbd_end_fit,
                                    bins = par.bins_R2,
                                    style = par.style_R2,
                                    save = True,
                                    robust = True)
                                    
    elif choice_shift == "Rayleigh" :
        d_lbd_Si = rhp.set_rayleigh_peak_to_0(da_full_spectrum_interp,
                           file_init,
                           'Rayleigh', 
                           tol_R2 = par.R_cSi, 
                           lbd_deb_fit = -40,
                           lbd_end_fit = 40, 
                           lbd_end_flatten = 70,
                           bins = par.bins_R2, 
                           style = 'seaborn', 
                           save = True, 
                           robust = True )
    else:
        d_lbd_Si = 0
        


    # mise à jour des xarrays en affectant une dimension physique [µm] aux axes x et y et en décalant les longueurs d'onde
    Z= rhp.update_xarray([da_raw, da_flatten, da_cosmic, da_denoise,da_std, da_full_spectrum],
                      file_info ,
                      L_x_default = par.L_x,
                      L_y_default = par.L_y,
                      d_lbd_Si = d_lbd_Si)
    da_raw, da_flatten, da_cosmic, da_denoise,da_std, da_full_spectrum = Z


    # pickling
    rhp.pickle(path_save, filename,"write",
               da_raw,da_flatten,da_cosmic, da_denoise, da_std,
               da_full_spectrum, da_raw_interp, da_full_spectrum_interp)

    if "max Stat" in choice_method :
      # aggregate classes

      class_Si_tot = reduce(lambda x, y: dict(**x, **y), list([X for _ , X in class_Si.items() ]))
      dic_lbds_ref = {k: v for d in list([X for _ , X in dic_ref.items() ]) for k, v in d.items()}

      # full spectra

      dg = rhp.Stat_MAX(da_denoise,
                        threshold_peak = par.threshold_peak,
                        Ecart = 5,
                        prominence = 0.5,
                        height = 3,
                        dic_lbds_ref = dic_lbds_ref,
                        dic_class_Si = class_Si_tot,
                        lbd_min_plot = None,
                        lbd_max_plot = None,
                        save = True,
                        show = True)

      # zooms

      dg = rhp.zoom_Stat_MAX(da_denoise,
                        threshold_peak = par.threshold_peak,
                        Ecart = 5,
                        prominence = 0.5,
                        height = 3,
                        dic_lbds_ref = dic_lbds_ref,
                        dic_class_Si = class_Si_tot,
                        lbd_min_plot = par.lbd_min_plot,
                        lbd_max_plot = par.lbd_max_plot,
                        save = True,
                        show = False)

    # phase imaging
    if "Phase imaging" in choice_method:
      dic_image_phase, dic_dic_phase_max, df_lbd = rhp.classe_phase_maximums_imaging(da_denoise,
                                                                                    class_Si,
                                                                                    dic_ref,
                                                                                    Ecart = 5,
                                                                                    prominence = 0.5,
                                                                                    height = 3,
                                                                                    Norme = par.Norme,
                                                                                    robust = par.robust,
                                                                                    save = True,
                                                                                    cmap = 'Greys',
                                                                                    path_image = path_image)

      # select spectra

      while True:

        da = rhp.choose_phase_xarray(list(df_lbd.index), dic_image_phase, dic_dic_phase_max)
        if da is None : break
        dpick = rhp.interactive_plot(da, da_cosmic,path_image , percentile = 100, save = True)

    if "PCA" in choice_method:
      rhp.Raman_PCA(da_denoise, k_PCA = 6)

    if "NMF" in choice_method:
      rhp.Raman_NMF(da_denoise, k_NMF = 6)

    if "robust PCA" in choice_method:
     dX,dY = rhp.Raman_Robust_PCA(da_flatten)

if __name__ == "__main__":
    phase_si()