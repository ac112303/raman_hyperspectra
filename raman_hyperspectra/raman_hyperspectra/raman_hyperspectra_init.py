def init_param( file ):

    '''
    parsing of the  'Paramètres de traitements' sheet of the Excel initialization file (called by init_rph)
            INPUT
                file : absolute path of the Excel initialization file
            OUTPUT
                par : objet which attributes are the parameters
    '''

    import pandas as pd

    class param(object):
        pass
    par = param()


    dm = pd.read_excel(file,
                 sheet_name= 'Paramètres de traitements')
    for row in dm.iterrows():
        setattr(par,row[1]['Nom de variable '],row[1]['Valeur'])

    return par

def init_rph( file,user_path ):

    '''
    parse the excel initialisation file
    INPUT
        file : name of the Excel file
        user_path : nae of path of the directory containing the initialization Excel file
    OUTPUT
        dic_ref : dictionnary {phase : reference wavelength}
        class_Si : dictionnary of dictionnaries of the bins upper and lower boundaries {classe name (A,B,C,..) : {phase classe :[lower boundary, upper boundary]} }
        dic_bornes : for future use
        par : objet with attributes the parameters (ex par.lbd_min the minimum wavelength to process the spectra)
        file_raw : absolute name of the Raman spectra/hyperspectra
        file_info : absoute name of the Raman file info prvided by Witec
        path_save : path of th directory to store the results
        file_image : absolute path of the topographic image of the sample

    '''


    import pandas as pd
    import numpy as np
    import os
    dialog = False

    par = init_param(file)

    for X in par.__dict__: # replace 'True' by logial True

        if par.__getattribute__(X) == 'True': setattr(par,X ,True)

    dm = pd.read_excel(file,
                       sheet_name= 'Phases',header = 1)

    # d_lbd shift de longueur d'onde par rapport à celle de rérérence si_c_1

    d_lbd = list(dm[dm['Nom de phase'] == "si_c_1" ]['Position (cm-1)'])[0] - par.lbd_si_c_1

    par.lbd_min_plot = [float(y) - d_lbd  for y in par.lbd_min_plot.replace(",", ".").split(';')]
    par.lbd_max_plot = [float(y) - d_lbd  for y in par.lbd_max_plot.replace(",", ".").split(';')]

    dic_bornes = {X[0] :(X[1],X[2],X[3])
                        for X in zip(dm['Nom de phase'],
                                     np.array(dm['Borne inf (cm-1)']) - d_lbd,
                                     np.array(dm['Borne sup (cm-1)']) - d_lbd,
                                     dm['Méthode'] )}

    dic_ref = {}
    class_Si = {}

    for x in dm.groupby(['Classe']):

        dic_ref[x[0]] = {value['Position (cm-1)'] - d_lbd: value['Nom de phase'] for _, value in x[1].T.to_dict().items()}

        class_Si[x[0]] = {value['Nom de phase'] :[value['Position (cm-1)'] - d_lbd - value['Interval inférieur  (cm-1)'],
                                                  value['Position (cm-1)'] - d_lbd + value['Interval supérieur  (cm-1)']]
                           for _, value in x[1].T.to_dict().items()}

    dm = pd.read_excel(file, sheet_name= 'Chemins')

    dic_paths = {x[0]:x[1]  for x in zip(dm['Nom de variable'],dm[user_path] )}


    if dialog:

        file_raw = select_file(dic_paths['path_files_exp'],
                               file_extension = ("txt files","*.txt"),
                               title = "Select data file" )

        file_info = select_file(dic_paths['path_files_info'],
                                file_extension = ("txt files","*.txt"),
                                title = "Select info file")

        file_model = select_file(dic_paths['path_files_model'],
                                 file_extension = ("xlsx files","*.xlsx"),
                                 title = "Select model file")
        path_save = dic_paths['path_save']

    else:

        dm = pd.read_excel(file, sheet_name= 'Fichiers')
        dic_file = {x[0]:x[1]  for x in zip(dm['Nom de variable'],dm[user_path] )}
        file_raw =  os.path.join(dic_paths['path_files_exp'] , dic_file['file_exp'])
        file_info = os.path.join(dic_paths['path_files_info'], dic_file['file_info'] )
        file_image = os.path.join(dic_paths['path_image'], dic_file['file_image'] )

        path_save = dic_paths['path_save']


    return dic_ref, class_Si, dic_bornes, par, file_raw, file_info, path_save, file_image

def users_init_path() :
    '''
    define the name and path of the Excel initialization file
    INPUT

    OUTPUT
        file_init : name of the Excel initialization file
        path : path of directory containing file_init

    '''

    import platform

    if platform.node()=='GRE040520':
        file_init = "C:\\Users\\ac112303\\Documents\\AC-Raman\\Fit_Raman\\Interface Raman_hyperspectra_V3.xlsx"
        user_path = 'Cas AC'

    elif platform.node()=='DESKTOP-MACMK58':
        file_init = "c:\\Temp\\Interface Raman_hyperspectra_V3.xlsx"
        user_path = 'Cas FB'

    return file_init, user_path

def init_raman_hyperspectra( file_init, user_path ):

    '''
    initialization of hyperspectra
    INPUT:
        file_init : name of the Excel initialization file
        user_path : name of the path containing the file_init
    OUTPUT:
        dic_ref : dictionnary {phase : reference wavelength}
        class_Si : dictionnary of dictionnaries of the bins upper and lower boundaries {classe name (A,B,C,..) : {phase classe :[lower boundary, upper boundary]} }
        dic_bornes : for future use
        par : objet with attributes the parameters (ex par.lbd_min the minimum wavelength to process the spectra)
        file_raw : absolute name of the Raman spectra/hyperspectra
        file_info : absoute name of the Raman file info prvided by Witec
        path_save : path of th directory to store the results
        file_image : absolute path of the topographic image of the sample
    '''

    from datetime import date
    import os
    import shutil
    import numpy as np


    dic_ref, class_Si, dic_bornes, par, file_raw, \
    file_info, path_save, file_image = init_rph(file_init, user_path)

    # création du répertoire de dépôt des fichiers

    head, tail = os.path.split(file_raw)
    filename = os.path.splitext(os.path.basename(tail))[0]
    filename = filename + " " + date.today().strftime('%d_%m_%Y')
    name_dir = path_save + filename

    if os.path.isdir(name_dir):
        shutil.rmtree(name_dir)

    os.mkdir( name_dir)
    os.chdir( name_dir)

    return dic_ref, class_Si, dic_bornes, par, file_raw, file_info, path_save, file_image, filename
