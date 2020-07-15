"""
Gui allowing  to interact with an hyperspectra
3rd party dependencies :
numpy, xarray, matplotlib, scipy, raman_hyperspectra.raman_hyperspectra_read_files
color lambda
"""

__all__ = [
    "gui_hyperspectra_visualization",
    "blend_topo_phase",
    "gui_phase_visualization",
    "gui_init_param",
    "gui_tune_param",
]


def gui_hyperspectra_visualization(da, file_image):

    """
    gui to:
        - visualize the spectra of an hyperspectrum. The visualisation of (i) the raw data;
            (ii) the baseline; (iii) the flattened denoised spectra; (iv) rhe maxima locations
        - save the spectra and their maxima to .xlsx format
        - choose among 9 baseline extraction algorithms
        - choose among 3 kinds of filters

    Arguments:
        - da (xarray) : hyperspectrum obtained using the funtions: read_RAMAN_RENISHAW_txt_2D
            or read_RAMAN_WITEC_2D

        - file_image (string): full path of the topographic image. The topograpphic image must have be
            cropped to suit the ROI dimension (gwyddion can be used). The length and the size of the image
            are da.x and da.y  attributes

    """

    # Standard Library dependencies
    import tkinter
    from tkinter import messagebox
    import tkinter.font as tkFont
    import os

    # 3rd party dependencies
    import cv2
    import PIL.Image, PIL.ImageTk
    import numpy as np
    import pandas as pd
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg,
        NavigationToolbar2Tk,
    )
    from matplotlib.figure import Figure
    from matplotlib.backend_bases import key_press_handler
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
    import matplotlib.pyplot as plt
    from pandastable import Table
    from scipy.signal import find_peaks

    # Internal dependencies
    from ..raman_hyperspectra_read_files import construct_xarray_0D
    from ..raman_hyperspectra_baseline import flatten_spectra
    from ..raman_hyperspectra_noise import spectra_noise_estimation_0D
    from ..raman_hyperspectra_denoise import smooth_spectrum

    # Parameter fpr fin_peaks
    PROMINENCE_FACTOR = 10
    WIDTH_PEAK = 3
    DISTANCE_PEAK = 5

    # initialisation of the menus
    LINEBASE = {
        "linear": ("linear", 400, 10),
        "top hat": ("top hat", 440),
        "rubberband": ("rubberband",),
        "drPLS": ("drPLS", 1.0e6, 0.5, 0.001, 100),
        "arPLS": ("arPLS", 1.0e5, 1.0e-3, 40),
        "ials ": ("ials", 10000, 0.01, 0.001),
        "als": ("als", 1.0e5, 1.0e-3, 40),
        "Modpoly": ("Modpoly", 2),
        "Imodpoly": ("Imodpoly", 2),
    }
    TYPE_LINEBASE = list(LINEBASE.keys())

    FILTERS = {
        "Savitzky-Golay (11,3)": ("Savitzky-Golay", 11, 3),
        "Savitzky-Golay (9,3)": ("Savitzky-Golay", 9, 3),
        "Savitzky-Golay (7,3)": ("Savitzky-Golay", 7, 3),
        "Savitzky-Golay (5,3)": ("Savitzky-Golay", 5, 3),
        "Savitzky-Golay (11,2)": ("Savitzky-Golay", 11, 2),
        "Savitzky-Golay (9,2)": ("Savitzky-Golay", 9, 2),
        "Savitzky-Golay (7,2)": ("Savitzky-Golay", 7, 2),
        "Savitzky-Golay (5,2)": ("Savitzky-Golay", 5, 2),
        "NRWT (1)": ("NRWT", 4, "db4", "soft", 1),
        "NRWT (0.9)": ("NRWT", 4, "db4", "soft", 0.9),
        "NRWT (0.8)": ("NRWT", 4, "db4", "soft", 0.8),
        "NRWT (0.7)": ("NRWT", 4, "db4", "soft", 0.7),
        "NRWT (0.6)": ("NRWT", 4, "db4", "soft", 0.6),
        "NRWT (0.5)": ("NRWT", 4, "db4", "soft", 0.5),
        "NRWT (0.4)": ("NRWT", 4, "db4", "soft", 0.4),
        "NRWT (0.3)": ("NRWT", 4, "db4", "soft", 0.3),
        "NRWT (0.25)": ("NRWT", 4, "db4", "soft", 0.25),
        "Moving Average (3)": ("Moving Average", 3),
        "Moving Average (5)": ("Moving Average", 5),
    }
    TYPE_FILTERS = list(FILTERS.keys())

    img_topo = cv2.cvtColor(cv2.imread(file_image), cv2.COLOR_BGR2RGB)
    HEIGHT, WIDTH, no_channels = img_topo.shape

    nbr_pixel = lambda x: WIDTH * x / (da.x.values[-1] - da.x.values[0])

    def save():

        """
        save the last selectected spectra in an .xlsx book with two sheets
            - the first sheet labeled Spectra contains 5 columns lbd|raw data|baseline|flatte|denoise
            - the second sheet labeled Maxima contains two columns lbd_max|peak amplitude
        """
        print(idx, idy)
        try:
            dic = {
                "lbd": lbd,
                "raw data": z_raw,
                "baseline": baseline,
                "flatten": z,
                "denoise": z_smooth,
            }
            df = pd.DataFrame.from_dict(dic)

            dic1 = {"lbd_max": lbd[idx_max], "peak amplitude": z_smooth[idx_max]}
            df1 = pd.DataFrame.from_dict(dic1)

            file = os.path.join(r"C:\Temp", f"{idx}_{idy}.xlsx")
            writer = pd.ExcelWriter(file)
            df.to_excel(writer, "Spectra")
            df1.to_excel(writer, "Maxima")
            writer.save()

            tkinter.messagebox.showinfo(
                title=None, message="Your file has been save under: " + file
            )

        except:  # before selecting a spectrum
            messagebox.showerror("Save function", "no data to save")

    def info():
        text = f"""
            hyperspectra name: {da.attrs['hyperspectra_name']}
                spectrum type : {da.lbd.attrs['spectrum type'] }
                # rows: {da.shape[1] }
                # columns: {da.shape[0] }

            wavelength:
                unit: {da.lbd.attrs['units']}
                shift correction: {da.lbd.attrs['shift correction'] }
                # wavelengths: {da.shape[2] }
                spectral resolution: {da.lbd.attrs['spectral_resolution']}

            topographic image :
            name : {os.path.basename(file_image)}
                height: {max(da.x.values)-min(da.x.values)} {da.x.attrs['units']}
                width : {max(da.y.values)-min(da.y.values)} {da.y.attrs['units']}
                height: {HEIGHT} pixels
                width: {WIDTH} pixels
            """
        messagebox.showinfo("hyperspectra info", text)

    def show():

        """
        show the last selectected pixel in an .xlsx file with two columns 'lbd','raw data'

        """
        try:
            if window1.state == "normal":
                window1.focus()
        except NameError as e:
            window1 = tkinter.Toplevel()
            window1.geometry("600x600+50+20")
            dic = {
                "lbd": lbd,
                "raw data": z_raw,
                "baseline": baseline,
                "flatten": z,
                "denoise": z_smooth,
            }
            df = pd.DataFrame.from_dict(dic)
            pt = Table(window1, dataframe=df)
            pt.show()

    def show_max():

        """
        show the last selectected pixel in an .xlsx file with two columns 'lbd','raw data'

        """
        try:
            if window1.state == "normal":
                window1.focus()
        except NameError as e:
            window1 = tkinter.Toplevel()
            window1.geometry("600x600+50+20")
            dic1 = {"lbd_max": lbd[idx_max], "peak amplitude": z_smooth[idx_max]}
            df1 = pd.DataFrame.from_dict(dic1)
            pt = Table(window1, dataframe=df1)
            pt.show()

    def create_circle(x, y, radius, canvasName):

        """
        draw a circle in a canvas. The upper left corner of the canvas is x0,y0

        Arguments:
         x (float): abcissa of the center of the cicle (pixels)
         y (float): ordinate of the center of the cicle (pixel)
         radius : radius of the circle
        """
        WIDTH_CIRCLE = 3

        x0 = x - radius
        y0 = y - radius
        x1 = x + radius
        y1 = y + radius
        return canvasName.create_oval(x0, y0, x1, y1, width=WIDTH_CIRCLE, outline="red")

    def create_square(x, y, square_side_length, canvasName):

        """
        draw a square in a canvas.

        Arguments:
         x (float): abcissa of the center of the square (pixels)
         y (float): ordinate of the center of the square (pixel)
         square_side_length : width of the square
        """

        WIDTH_PERIMETER = 3

        x0 = x - square_side_length / 2
        y0 = y + square_side_length / 2
        x1 = x + square_side_length / 2
        y1 = y - square_side_length / 2

        return canvasName.create_rectangle(
            x0, y0, x1, y1, width=WIDTH_PERIMETER, outline="blue"
        )

    def create_scale_bar(canvasName, length_line):

        """
        Draw a scale bar

        Arguments:
            canvasName : name of the canvas where the square is drawn
            length (real): length_line in physical unit

        """

        COLOR = "blue"
        WIDTH_BAR = 3

        bar_scale = canvasName.create_line(
            WIDTH - nbr_pixel(5),
            HEIGHT - nbr_pixel(2),
            WIDTH - nbr_pixel(5) - nbr_pixel(length_line),
            HEIGHT - nbr_pixel(2),
            fill=COLOR,
            width=WIDTH_BAR,
        )

        right_end = canvasName.create_line(
            WIDTH - nbr_pixel(5),
            HEIGHT - nbr_pixel(1.5),
            WIDTH - nbr_pixel(5),
            HEIGHT - nbr_pixel(2.5),
            fill=COLOR,
            width=WIDTH_BAR,
        )

        left_scale = canvasName.create_line(
            WIDTH - nbr_pixel(5) - nbr_pixel(length_line),
            HEIGHT - nbr_pixel(2.5),
            WIDTH - nbr_pixel(5) - nbr_pixel(length_line),
            HEIGHT - nbr_pixel(1.5),
            fill=COLOR,
            width=WIDTH_BAR,
        )

        canvasName.create_text(
            WIDTH - nbr_pixel(5) - int(nbr_pixel(length_line) / 2),
            HEIGHT - nbr_pixel(3.3),
            text=str(length_line) + " µm",
            font=FONTSTYLE,
            fill=COLOR,
        )
        return bar_scale

    def refresh(circle_list):

        for objet in circle_list:
            canvas1.delete(objet)

    def clavier(event):

        """
        treatment of the left click event:
            - first click: acqisition of the coordinates (x0,y0) of the upper left corner of the canvas
            - subsequent click : plot the spectra of the selected

        """

        global idx, idy, lbd, z_raw, baseline, z, z_smooth, idx_max
        clavier.flag = getattr(clavier, "flag", True)

        if clavier.flag:  # First call
            x0, y0 = canvas1.winfo_pointerxy()
            clavier.x0 = getattr(clavier, "x0", x0)
            clavier.y0 = getattr(clavier, "y0", y0)
            clavier.flag = False
            text.set("left click to select an image point")
            label1 = tkinter.Label(frame_img, textvariable=text, font=FONTSTYLE)
            label1.grid(row=1, column=0)

        else:
            x, y = canvas1.winfo_pointerxy()

            c = create_circle(x - clavier.x0, y - clavier.y0, 5, canvas1)
            circle_list.append(c)
            save_button["state"] = tkinter.NORMAL
            show_button["state"] = tkinter.NORMAL
            show_max["state"] = tkinter.NORMAL

            idx = min(int(da.shape[0] * (x - clavier.x0) / WIDTH), da.shape[0] - 1)
            idy = min(int(da.shape[1] * (y - clavier.y0) / HEIGHT), da.shape[1] - 1)
            x_scale = da.x.values[idx]
            y_scale = da.y.values[idy]
            unit_x = da.x.attrs["units"]
            unit_y = da.y.attrs["units"]
            text.set(
                f"idx = {idx}, idy = {idy}, x = {x_scale} {unit_x}, y = {y_scale} {unit_y}"
            )
            label1 = tkinter.Label(frame_img, textvariable=text, font=FONTSTYLE)
            label1.grid(row=1, column=0)

            baseline_choice = type_baseline.get()
            filter_choice = type_filter.get()

            fig = Figure(figsize=(5, 4), dpi=100)
            ax = fig.add_subplot(1, 1, 1)

            lbd = da.lbd.values
            z_raw = da[idy, idx].values
            db = construct_xarray_0D(z_raw, lbd)
            z, baseline = flatten_spectra(db, LINEBASE[baseline_choice])

            if raw_data.get() == 1:
                ax.plot(lbd, da.isel(x=idx, y=idy), label="raw")

            if draw_baseline.get() == 1:
                ax.plot(lbd, z, lbd, baseline)

            ax.plot(lbd, z)
            dc = construct_xarray_0D(z, lbd)
            std_gauss, std_sg, *_ = spectra_noise_estimation_0D(
                dc, ("Gauss", 1.7), ("Savitzky-Golay", 9, 3), lbd_dep=100, lbd_end=480,
            )

            z_smooth = smooth_spectrum(z, FILTERS[filter_choice], std_gauss)
            ax.plot(lbd, z, lbd, z_smooth)

            idx_max_tot, info_peak = find_peaks(
                z_smooth,
                distance=DISTANCE_PEAK,
                prominence=PROMINENCE_FACTOR * std_sg,
                width=WIDTH_PEAK,
            )
            idx_max = list(idx_max_tot)

            if plot_max.get() == 1:
                ax.plot(lbd[idx_max], z_smooth[idx_max], "or")

            ax.set_xlabel("Wavelength [cm-1]")
            ax.set_title(f"{baseline_choice},  {filter_choice}")

            frame_plot = tkinter.LabelFrame(window)
            frame_plot.grid(row=0, column=1)
            canvas2 = FigureCanvasTkAgg(fig, master=frame_plot)
            canvas2.draw()
            canvas2.get_tk_widget()
            canvas2.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

            toolbar = NavigationToolbar2Tk(canvas2, frame_plot)
            toolbar.update()
            canvas2.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

            label4 = tkinter.Label(
                frame_plot, text=f"    spectra std:{std_gauss:.2f} ", font=FONTSTYLE
            )
            label4.pack()

    def slide(val):

        global idx, idy, lbd, z_raw, baseline, z, z_smooth, idx_max

        idx = slide_idx.get()
        idy = slide_idy.get()

        slide.flag = getattr(slide, "flag", True)

        if (
            slide.flag
        ):  # First call we pick coordinate (x0,y0) of the upper left corner of the canvas
            slide.flag = False
            r = create_square(
                WIDTH * idx / (len(da.x) - 1),
                HEIGHT * idy / (len(da.y) - 1),
                10,
                canvas1,
            )
            slide.r = getattr(slide, "r", r)
            save_button["state"] = tkinter.NORMAL
            show_button["state"] = tkinter.NORMAL
            show_max["state"] = tkinter.NORMAL
        else:
            canvas1.delete(slide.r)
            r = create_square(
                WIDTH * idx / (len(da.x) - 1),
                HEIGHT * idy / (len(da.y) - 1),
                10,
                canvas1,
            )
            slide.r = r

        baseline_choice = type_baseline.get()
        filter_choice = type_filter.get()

        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(1, 1, 1)

        lbd = da.lbd.values
        z_raw = da[idy, idx].values
        db = construct_xarray_0D(z_raw, lbd)
        z, baseline = flatten_spectra(db, LINEBASE[baseline_choice])

        if raw_data.get() == 1:
            ax.plot(lbd, da.isel(x=idx, y=idy), label="raw")

        if draw_baseline.get() == 1:
            ax.plot(lbd, z, lbd, baseline)

        ax.plot(lbd, z)
        dc = construct_xarray_0D(z, da.lbd, tool="WITEK")
        std_sg, std_gauss, *_ = spectra_noise_estimation_0D(
            dc, ("Gauss", 1.7), ("Savitzky-Golay", 9, 3), lbd_dep=100, lbd_end=480,
        )

        z_smooth = smooth_spectrum(z, FILTERS[filter_choice], std_sg)
        ax.plot(lbd, z, lbd, z_smooth)

        idx_max_tot, info_peak = find_peaks(
            z_smooth,
            distance=DISTANCE_PEAK,
            prominence=PROMINENCE_FACTOR * std_sg,
            width=WIDTH_PEAK,
        )
        idx_max = list(idx_max_tot)

        if plot_max.get() == 1:
            ax.plot(lbd[idx_max], z_smooth[idx_max], "or")

        ax.set_xlabel("Wavelength [cm-1]")
        ax.set_title(f"{baseline_choice},  {filter_choice}")

        frame_plot = tkinter.LabelFrame(window)
        frame_plot.grid(row=0, column=1)
        canvas2 = FigureCanvasTkAgg(fig, master=frame_plot)
        canvas2.draw()
        canvas2.get_tk_widget()
        canvas2.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas2, frame_plot)
        toolbar.update()
        canvas2.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        label4 = tkinter.Label(
            frame_plot, text=f"    spectra std:{std_gauss:.2f} ", font=FONTSTYLE
        )
        label4.pack()

    window = tkinter.Tk()

    window.title("Hyperspectra visualization")
    window.geometry(f"{int(2.4*WIDTH)}x{HEIGHT+100}+0+0")
    window.resizable(width=False, height=False)

    FONTSTYLE = tkFont.Font(family="Lucida Grande", size=13, weight="bold")

    text = tkinter.StringVar()
    circle_list = []

    # Create Canvas for image
    frame_img = tkinter.LabelFrame(window)
    frame_img.grid(row=0, column=0)

    canvas1 = tkinter.Canvas(frame_img, width=WIDTH, height=HEIGHT, bd=0)
    canvas1.grid(row=0, column=0)

    text.set("left click on the upper left image corner")
    label1 = tkinter.Label(
        frame_img, textvariable=text, font=FONTSTYLE, foreground="red"
    )
    label1.grid(row=1, column=0)

    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img_topo))

    canvas1.create_image(0, 0, image=photo, anchor=tkinter.NW)
    canvas1.configure(bd=0, cursor="circle")
    create_scale_bar(canvas1, 10)

    canvas1.bind("<Button-1>", clavier)

    # frame method
    # baseline correction method
    frame_menu = tkinter.Frame(window)
    frame_menu.grid(row=2, column=0)
    label2 = tkinter.Label(frame_menu, text="baseline", font=FONTSTYLE)
    label2.grid(row=0, column=0)

    type_baseline = tkinter.StringVar()
    type_baseline.set(TYPE_LINEBASE[1])
    drop = tkinter.OptionMenu(frame_menu, type_baseline, *TYPE_LINEBASE)
    drop.config(font=FONTSTYLE)
    drop.grid(row=0, column=1)

    # filter choice
    label3 = tkinter.Label(frame_menu, text="    filter", font=FONTSTYLE)
    label3.grid(row=0, column=2)

    type_filter = tkinter.StringVar()
    type_filter.set(TYPE_FILTERS[0])
    drop_filter = tkinter.OptionMenu(frame_menu, type_filter, *TYPE_FILTERS)
    drop_filter.config(font=FONTSTYLE)
    drop_filter.grid(row=0, column=3)

    frame_option = tkinter.Frame(window)
    frame_option.grid(row=0, column=2)

    # plot max
    plot_max = tkinter.IntVar()
    plot_max_check = tkinter.Checkbutton(
        frame_option, text="plot maxima", variable=plot_max, font=FONTSTYLE
    )
    plot_max_check.pack(side=tkinter.TOP, anchor=tkinter.W)

    # plot raw data
    raw_data = tkinter.IntVar()
    raw_data_check = tkinter.Checkbutton(
        frame_option, text="plot raw data", variable=raw_data, font=FONTSTYLE
    )
    raw_data_check.pack(side=tkinter.TOP, anchor=tkinter.W)

    # plot baseline
    draw_baseline = tkinter.IntVar()
    draw_baseline_data_check = tkinter.Checkbutton(
        frame_option, text="plot baseline", variable=draw_baseline, font=FONTSTYLE
    )
    draw_baseline_data_check.pack(side=tkinter.TOP, anchor=tkinter.W)

    BD = 4
    WIDTH_BUTTON = 10
    PADY = 5
    # save button
    save_button = tkinter.Button(
        frame_option,
        text="Export",
        width=WIDTH_BUTTON,
        pady=PADY,
        bd=BD,
        command=save,
        font=FONTSTYLE,
        state=tkinter.DISABLED,
    )
    save_button.pack()

    # show button
    show_button = tkinter.Button(
        frame_option,
        text="Show data",
        width=WIDTH_BUTTON,
        pady=PADY,
        bd=BD,
        command=show,
        font=FONTSTYLE,
        state=tkinter.DISABLED,
    )
    show_button.pack()

    # show max
    show_max = tkinter.Button(
        frame_option,
        text="Show MAXs",
        width=WIDTH_BUTTON,
        pady=PADY,
        bd=BD,
        command=show_max,
        font=FONTSTYLE,
        state=tkinter.DISABLED,
    )
    show_max.pack()

    # info button
    info_button = tkinter.Button(
        frame_option,
        text="Info",
        command=info,
        width=WIDTH_BUTTON,
        pady=PADY,
        bd=BD,
        font=FONTSTYLE,
    )
    info_button.pack()

    # refresh command=lambda: refresh(circle_list)
    refresh_button = tkinter.Button(
        frame_option,
        text="Refresh",
        width=WIDTH_BUTTON,
        pady=PADY,
        bd=BD,
        command=lambda: refresh(circle_list),
        font=FONTSTYLE,
    )
    refresh_button.pack()

    # exit button
    tkinter.Button(
        frame_option,
        text="Quit",
        width=WIDTH_BUTTON,
        pady=PADY,
        bd=BD,
        command=window.destroy,
        font=FONTSTYLE,
        foreground="red",
    ).pack()
    # slides
    slide_idx = tkinter.Scale(
        frame_option,
        orient="horizontal",
        from_=0,
        to=len(da.x) - 1,
        resolution=1,
        length=150,
        label="idx",
        font=FONTSTYLE,
        command=slide,
    )
    slide_idx.pack()
    slide_idy = tkinter.Scale(
        frame_option,
        orient="horizontal",
        from_=0,
        to=len(da.y) - 1,
        resolution=1,
        length=150,
        label="idy",
        font=FONTSTYLE,
        command=slide,
    )
    slide_idy.pack()

    window.mainloop()


def blend_topo_phase(
    file_image,
    dic_image_phase,
    dic_dic_phase_max,
    cmap_topo="bone",
    cmap_phase="hot",
    alpha=0.3,
):

    """
    Blending of the topographic and a phase image.
    The transparency alpha of the topographic image is set to 0 whhere the phase is present and
    to 1 elsewhere.

    Arguments:
        - file_image (string): full path of the topographic image
        - dic_image (dict of dict): {classe : {phase: [idx_x max, idx y max, lbd_max,height, ] }} built
        by the function classe_phase_maximums_imaging
        - dic_dic_phase_max (dict): dictionary of dictionaries
                            {classe : {phase: [idx_x max, idx y max, lbd_max,height, ] }}
        - cmap_topo : cmap for the topographic image (default cmap_topo = 'bone')
        - cmap_phase : cmap for the phase image (default cmap_phase = 'hot')
    """

    def blend(file_image, db, name_phase, cmap_topo="bone", cmap_phase="hot"):
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.interpolate import RegularGridInterpolator
        import os

        def rgb2gray(rgb):

            """
            rgb to gray conversion conversion using  CCIR 601 norme
            see https://en.wikipedia.org/wiki/Luma_(video)
            Arguments:
                rgb (nparray): [N,M,3] for rbg; [N,M,3,1] for rgba
            Returns:
                gray (nparray): [N.M] array
            """
            gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

            return gray

        def interpol_image():

            """
            Resize and interpol the phase image onto the topographic image.
            We implicitly assume that the resolution, in pixels, of the topographic image
            is greater that phase image one
            """

            METHOD = "nearest"  # "linear" could be used
            n_row_topo = img_topo.shape[1]
            n_col_topo = img_topo.shape[0]
            n_row_phase = img_phase.shape[0]
            n_col_phase = img_phase.shape[1]

            interp_func = RegularGridInterpolator(
                (
                    np.linspace(0, n_col_phase - 1, n_col_phase),
                    np.linspace(0, n_row_phase - 1, n_row_phase),
                ),
                img_phase,
                method=METHOD,
            )

            # buid the list of coordinates [x,y] of of points where phase_interp is interpolated
            pts = [
                [x, y]
                for x in np.linspace(0, n_col_topo - 1, n_col_topo)
                * (n_col_phase - 1)
                / (n_col_topo - 1)
                for y in np.linspace(0, n_row_topo - 1, n_row_topo)
                * (n_row_phase - 1)
                / (n_row_topo - 1)
            ]

            phase_interp = interp_func(pts).reshape((n_col_topo, n_row_topo))

            return phase_interp

        PERCENTILE = 98  # use to trim the outliers pertaining to the two last centiles
        img_topo = plt.imread(file_image)
        img_topo = rgb2gray(img_topo)

        A = db.values.reshape((db.values.shape[0], db.values.shape[1]))
        img_phase = A[
            ::-1, :
        ]  # x-reflection to be compatible with the topographic image

        threshold = np.percentile(img_phase, PERCENTILE)
        img_phase[np.where(img_phase > threshold)] = threshold

        fig = plt.figure(figsize=(8, 8))
        if threshold > 0:
            phase_interp = (
                interpol_image()
            )  # set topographic and phase image to the same size in pixels
            phase_image = 255 * phase_interp / np.max(phase_interp)
            topo_image = 255 * img_topo / np.max(img_topo)
            plt.imshow(phase_image, cmap=cmap_phase)
            mask = np.where(phase_image < 40, alpha, 0.0)
            plt.imshow(topo_image, alpha=mask, cmap=cmap_topo)
        else:
            topo_image = 255 * img_topo / np.max(img_topo)
            plt.imshow(topo_image, cmap=cmap_topo, alpha=alpha)
        plt.axis("off")

        save_file = os.path.join(r"c:\Temp", f"blend_topo_phase_{name_phase}.png")
        fig.savefig(save_file, pad_inches=0, bbox_inches="tight")

        return

    phase_unique = []  # unwrap dic_image_phase to build the list of phases
    for _, values in dic_image_phase.items():
        for phase in values.phase.values:
            if phase != "Image":
                phase_unique.append(phase.split("(")[0])

    for name_phase in list(set(phase_unique)):

        # find the name_phase classe
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
        blend(file_image, da, name_phase.split("(")[0], cmap_topo, cmap_phase)

    return


def gui_phase_visualization(da, dic_ref=None):

    """
    gui to:
        - visualize the spectra associated to the phases of an hyperspectrum.
        - save the spectra .xlsx format

    Arguments:
        - da (xarray) : hyperspectrum
        - dic_ref: dict pf dict {classe:{lbd ref: phase,...},...} (default = None)
    """

    # Standard Library dependencies
    import tkinter
    from tkinter import messagebox
    import tkinter.font as tkFont
    from itertools import cycle
    import os
    from collections import ChainMap

    # 3rd party dependencies
    import PIL.Image, PIL.ImageTk
    import numpy as np
    import pandas as pd
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg,
        NavigationToolbar2Tk,
    )
    from matplotlib.figure import Figure
    from matplotlib.backend_bases import key_press_handler
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
    import matplotlib.pyplot as plt
    from pandastable import Table

    # find the .png files blending topograpy and phase
    list_file_phases = {
        file[17:-4]: file
        for file in os.listdir(r"c:\Temp")
        if file.endswith(".png") and file.startswith("blend_topo_phase")
    }
    list_phases = list(list_file_phases.keys())

    nbr_pixel = lambda x: WIDTH * x / (da.x.values[-1] - da.x.values[0])

    def phase_change():

        """
        Upgrade the phase image in the canvas1 whenever a change of phase is detected

        """

        phase = type_phase.get()
        canvas1.itemconfig(phase_img, image=my_images[phase])

    def info():
        text = f"""
                hyperspectra name: {da.attrs['hyperspectra_name']}
                    spectrum type : {da.lbd.attrs['spectrum type'] }
                    # rows: {da.shape[1] }
                    # columns: {da.shape[0] }

                wavelength:
                    unit: {da.lbd.attrs['units']}
                    shift correction: {da.lbd.attrs['shift correction'] }
                    # wavelengths: {da.shape[2] }
                    spectral resolution: {da.lbd.attrs['spectral_resolution']}

                topographic image :
                    height: {max(da.x.values)-min(da.x.values)} {da.x.attrs['units']}
                    width : {max(da.y.values)-min(da.y.values)} {da.y.attrs['units']}
                    height: {HEIGHT} pixels
                    width: {WIDTH} pixels
                """
        messagebox.showinfo("hyperspectra info", text)

    def show():

        """
        show the last selectected pixel in an .xlsx file with two columns 'lbd','raw data'

        """
        try:
            if window1.state == "normal":
                window1.focus()
        except NameError as e:
            window1 = tkinter.Toplevel()
            window1.geometry("600x600+50+20")
            dic = {"lbd": lbd, "raw data": z_raw}
            df = pd.DataFrame.from_dict(dic)
            pt = Table(window1, dataframe=df)
            pt.show()

    def save():

        """
        save the last selectected pixel in an .xlsx file with two columns 'lbd','raw data'

        """
        try:
            dic = {"lbd": lbd, "raw data": z_raw}
            df = pd.DataFrame.from_dict(dic)
            file = os.path.join(r"C:\Temp", f"{type_phase.get()}_{idx}_{idy}.xlsx")
            df.to_excel(file)
            file = os.path.join(r"C:\Temp", f"{type_phase.get()}_{idx}_{idy}.csv")
            df.to_csv(file)
            tkinter.messagebox.showinfo(
                title=None, message="Your file has been save under: " + file
            )
        except:
            messagebox.showerror("Save function", "no data to save")

    def create_circle(x, y, radius, canvas_name):

        """
        draw a circle in a canvas. The upper left corner of the canvas is x0,y0

        Arguments:
         x (float): abcissa of the center of the cicle (pixels)
         y (float): ordinate of the center of the cicle (pixel)
         radius : radius of the circle (the color is define by the dict phase_color[phase])
        """

        WIDTH_CIRCLE = 3

        x0 = x - radius
        y0 = y - radius
        x1 = x + radius
        y1 = y + radius

        return canvas_name.create_oval(
            x0, y0, x1, y1, width=WIDTH_CIRCLE, outline=phase_color[type_phase.get()]
        )

    def create_square(x, y, square_side_length, canvasName):

        """
        draw a square in a canvas.

        Arguments:
         x (float): abcissa of the center of the square (pixels)
         y (float): ordinate of the center of the square (pixel)
         square_side_length : width of the square
        """

        WIDTH_PERIMETER = 3

        x0 = x - square_side_length / 2
        y0 = y + square_side_length / 2
        x1 = x + square_side_length / 2
        y1 = y - square_side_length / 2

        return canvasName.create_rectangle(
            x0, y0, x1, y1, width=WIDTH_PERIMETER, outline=phase_color[type_phase.get()]
        )

    def create_scale_bar(canvasName, length_line):

        """
        Draw a scale bar

        Arguments:
            canvasName : name of the canvas where the square is drawn
            length (real): length_line in physical unit (µm) or in pixel (u.a.)

        """

        COLOR = "blue"
        WIDTH_BAR_SCALE = 3

        bar_scale = canvasName.create_line(
            WIDTH - nbr_pixel(5),
            HEIGHT - nbr_pixel(2),
            WIDTH - nbr_pixel(5) - nbr_pixel(length_line),
            HEIGHT - nbr_pixel(2),
            fill=COLOR,
            width=WIDTH_BAR_SCALE,
        )

        right_end = canvasName.create_line(
            WIDTH - nbr_pixel(5),
            HEIGHT - nbr_pixel(1.5),
            WIDTH - nbr_pixel(5),
            HEIGHT - nbr_pixel(2.5),
            fill=COLOR,
            width=WIDTH_BAR_SCALE,
        )

        left_scale = canvasName.create_line(
            WIDTH - nbr_pixel(5) - nbr_pixel(length_line),
            HEIGHT - nbr_pixel(2.5),
            WIDTH - nbr_pixel(5) - nbr_pixel(length_line),
            HEIGHT - nbr_pixel(1.5),
            fill=COLOR,
            width=WIDTH_BAR_SCALE,
        )

        canvasName.create_text(
            WIDTH - nbr_pixel(5) - int(nbr_pixel(length_line) / 2),
            HEIGHT - nbr_pixel(3.3),
            text=str(length_line) + " " + da.x.attrs["units"],
            font=FONTSTYLE,
            fill=COLOR,
        )

    def refresh(circle_list):

        """
        Erase all the cicles drawn on the cancas1
        Arguments:
            cicle_list (list of int): list of the objets to be erased
        """

        for objet in circle_list:
            canvas1.delete(objet)

    def slide(val):

        """
        treatment of the slide event (i) draw a square marker, plot the spectume
        for the idx, idy slide values

        """

        global idx, idy, lbd, z_raw

        SQUARE_SIDE_LENGTH = 10
        idx = slide_idx.get()
        idy = slide_idy.get()

        slide.flag = getattr(slide, "flag", True)

        if slide.flag:  # First call we draw a square and memorize the object
            slide.flag = False
            square = create_square(
                WIDTH * idx / (len(da.x) - 1),
                HEIGHT * idy / (len(da.y) - 1),
                SQUARE_SIDE_LENGTH,
                canvas1,
            )
            slide.square = getattr(
                slide, "square", square
            )  # memorize the square object in a static variable
            save_button["state"] = tkinter.NORMAL
            show_button["state"] = tkinter.NORMAL

        else:
            canvas1.delete(slide.square)
            square = create_square(
                WIDTH * idx / (len(da.x) - 1),
                HEIGHT * idy / (len(da.y) - 1),
                SQUARE_SIDE_LENGTH,
                canvas1,
            )
            slide.square = square

        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(1, 1, 1)

        lbd = da.lbd.values
        z_raw = da[idy, idx].values

        ax.plot(lbd, z_raw)
        ax.set_xlabel("lbd (cm-1)")
        if Flag_dic_ref:
            ax.plot(
                [dic_phase_lbd[type_phase.get()]] * 2, [min(z_raw), max(z_raw)], "--"
            )
            ax.set_title(
                f"{type_phase.get()}: {dic_phase_lbd[type_phase.get()]} [cm-1]"
            )
        else:
            ax.set_title(f"{type_phase.get()}")

        frame_plot = tkinter.LabelFrame(window)
        frame_plot.grid(row=0, column=1)
        canvas2 = FigureCanvasTkAgg(fig, master=frame_plot)
        canvas2.draw()
        canvas2.get_tk_widget()
        canvas2.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas2, frame_plot)
        toolbar.update()
        canvas2.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    def clavier(event):

        """
        treatment of the left click event:
            - first click: acqisition of the coordinates (x0,y0) of the upper left corner of the canvas
            - subsequent click : if the name phase is changed efresh the canvas with the new image
                               : otherwise plot the spectra of the selected pixel

        Argument:
          event : <Button-1> left click

        """

        global idx, idy, lbd, z_raw, x0, y0
        clavier.flag = getattr(clavier, "flag", True)

        if (
            clavier.flag
        ):  # First call we pick coordinate (x0,y0) of the upper left corner of the canvas
            x0, y0 = canvas1.winfo_pointerxy()
            clavier.x0 = getattr(clavier, "x0", x0)
            clavier.y0 = getattr(clavier, "y0", y0)
            clavier.phase = getattr(clavier, "phase", type_phase.get())
            clavier.flag = False
            text.set("left click to select an image point")
            label1 = tkinter.Label(frame_img, textvariable=text, font=FONTSTYLE)
            label1.grid(row=1, column=0)

        else:
            save_button["state"] = tkinter.NORMAL
            show_button["state"] = tkinter.NORMAL

            x, y = canvas1.winfo_pointerxy()

            if clavier.phase != type_phase.get():  # check a change of phase
                phase_change()
                clavier.phase = type_phase.get()
                return

            circle = create_circle(x - clavier.x0, y - clavier.y0, 5, canvas1)
            circle_list.append(circle)

            idx = min(int(da.shape[0] * (x - clavier.x0) / WIDTH), da.shape[0] - 1)
            idy = min(int(da.shape[1] * (y - clavier.y0) / HEIGHT), da.shape[1] - 1)
            x_scale = da.x.values[idx]
            y_scale = da.y.values[idy]
            unit_x = da.x.attrs["units"]
            unit_y = da.y.attrs["units"]
            text.set(
                f"idx = {idx}, idy = {idy}, x = {x_scale} {unit_x}, y = {y_scale} {unit_y}"
            )
            label1 = tkinter.Label(frame_img, textvariable=text, font=FONTSTYLE)
            label1.grid(row=1, column=0)

            fig = Figure(figsize=(5, 4), dpi=100)
            ax = fig.add_subplot(1, 1, 1)

            lbd = da.lbd.values
            z_raw = da[idy, idx].values

            ax.plot(lbd, z_raw)
            ax.set_xlabel("lbd (cm-1)")
            if Flag_dic_ref:
                ax.plot(
                    [dic_phase_lbd[type_phase.get()]] * 2,
                    [min(z_raw), max(z_raw)],
                    "--",
                )
                ax.set_title(
                    f"{type_phase.get()}: {dic_phase_lbd[type_phase.get()]} [cm-1]"
                )
            else:
                ax.set_title(f"{type_phase.get()}")

            frame_plot = tkinter.LabelFrame(window)
            frame_plot.grid(row=0, column=1)
            canvas2 = FigureCanvasTkAgg(fig, master=frame_plot)
            canvas2.draw()
            canvas2.get_tk_widget()
            canvas2.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

            toolbar = NavigationToolbar2Tk(canvas2, frame_plot)
            toolbar.update()
            canvas2.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    window = tkinter.Tk()

    window.title("Hyperspectra visualization")
    circle_list = []

    Flag_dic_ref = False
    if dic_ref:
        Flag_dic_ref = True
        dic_phase_lbd = dict(
            ChainMap(*[dict(zip(a.values(), a.keys())) for a in dic_ref.values()])
        )

    phase_color = {}
    dic_color = {
        x[0]: x.lower()
        for x in ["white", "Black", "red", "green", "blue", "cyan", "yellow", "magenta"]
    }
    my_images = {}
    for phase, color in zip(list_phases, cycle("".join(dic_color.keys()))):
        my_images[phase] = tkinter.PhotoImage(
            file=os.path.join(r"c:\Temp", list_file_phases[phase])
        )
        phase_color[phase] = dic_color[color]

    image = PIL.Image.open(os.path.join(r"c:\Temp", list_file_phases[list_phases[0]]))
    WIDTH, HEIGHT = image.size
    FONTSTYLE = tkFont.Font(family="Lucida Grande", size=13, weight="bold")

    window.geometry(f"{int(2.6*WIDTH)}x{HEIGHT+100}+10+10")
    window.resizable(width=False, height=False)
    text = tkinter.StringVar()

    # Create Canvas for phase/topo image
    frame_img = tkinter.LabelFrame(window)
    frame_img.grid(row=0, column=0)

    canvas1 = tkinter.Canvas(frame_img, width=WIDTH, height=HEIGHT, bd=0)
    canvas1.grid(row=0, column=0)

    phase_img = canvas1.create_image(
        0, 0, image=my_images[list_phases[0]], anchor=tkinter.NW
    )

    text.set("left click on the upper left image corner")
    label1 = tkinter.Label(
        frame_img, textvariable=text, font=FONTSTYLE, foreground="red"
    )
    label1.grid(row=1, column=0)

    canvas1.configure(bd=0, cursor="circle")
    create_scale_bar(canvas1, 10)

    canvas1.bind("<Button-1>", clavier)

    # menu phase
    frame_menu = tkinter.Frame(window)
    frame_menu.grid(row=2, column=0)
    label2 = tkinter.Label(frame_menu, text="Phase", font=FONTSTYLE)
    label2.grid(row=0, column=0)

    type_phase = tkinter.StringVar()
    type_phase.set(list_phases[1])
    drop = tkinter.OptionMenu(frame_menu, type_phase, *list_phases)
    drop.config(font=FONTSTYLE)
    drop.grid(row=0, column=1)

    frame_option = tkinter.Frame(window)
    frame_option.grid(row=0, column=2)

    WIDTH_BUTTON = 10
    PADY = 5

    # save button
    save_button = tkinter.Button(
        frame_option,
        text="Export",
        width=WIDTH_BUTTON,
        pady=PADY,
        command=save,
        font=FONTSTYLE,
        state=tkinter.DISABLED,
    )
    save_button.pack()

    # show button
    show_button = tkinter.Button(
        frame_option,
        text="Show data",
        width=WIDTH_BUTTON,
        pady=PADY,
        command=show,
        font=FONTSTYLE,
        state=tkinter.DISABLED,
    )
    show_button.pack()

    # info button
    info_button = tkinter.Button(
        frame_option,
        text="Info",
        command=info,
        width=WIDTH_BUTTON,
        pady=PADY,
        font=FONTSTYLE,
    )
    info_button.pack()

    # refresh command=lambda: refresh(circle_list)
    refresh_button = tkinter.Button(
        frame_option,
        text="Refresh",
        width=WIDTH_BUTTON,
        pady=PADY,
        command=lambda: refresh(circle_list),
        font=FONTSTYLE,
    )
    refresh_button.pack()

    # exit button
    tkinter.Button(
        frame_option,
        text="Quit",
        width=WIDTH_BUTTON,
        pady=PADY,
        command=window.destroy,
        font=FONTSTYLE,
        foreground="red",
    ).pack()

    # slides
    slide_idx = tkinter.Scale(
        frame_option,
        orient="horizontal",
        from_=0,
        to=len(da.x) - 1,
        resolution=1,
        length=150,
        label="idx",
        font=FONTSTYLE,
        command=slide,
    )
    slide_idx.pack()
    slide_idy = tkinter.Scale(
        frame_option,
        orient="horizontal",
        from_=0,
        to=len(da.y) - 1,
        resolution=1,
        length=150,
        label="idy",
        font=FONTSTYLE,
        command=slide,
    )
    slide_idy.pack()

    window.mainloop()


def gui_init_param(file_model=r"c:\Temp\modele.csv", file_init=None, sheet=None):

    """
    Interactive set up of the Raman model

    Arguments:
        file_model (str): full path name of the .csv file where to save the model (default 'c:\Temp\modele.csv')
        dic_default (dict of dict): use to set model default values {type_of_funtion:{param1:value1,param2:value2,...},... }
    """

    # Standard Library dependencies
    import tkinter
    from tkinter import messagebox
    import tkinter.font as tkFont
    import re

    # 3rd party dependencies
    import numpy as np
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg,
        NavigationToolbar2Tk,
    )
    from matplotlib.figure import Figure
    from matplotlib.backend_bases import key_press_handler
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
    import matplotlib.pyplot as plt
    import pandas as pd

    # Internal dependencies
    from ..raman_hyperspectra_fit import init_model, sum_functions

    global idx_row, E

    noise = 0

    fenetre = tkinter.Tk()

    FONTSTYLE = tkFont.Font(family="Lucida Grande", size=10, weight="bold")

    frame_option = tkinter.Frame(fenetre)
    frame_option.grid(row=0, column=0)

    frame_table = tkinter.Frame(fenetre)
    frame_table.grid(row=1, column=0)

    frame_plot1 = tkinter.Frame(fenetre)
    frame_plot1.grid(row=2, column=0)

    frame_action_plot1 = tkinter.Frame(fenetre)
    frame_action_plot1.grid(row=2, column=1)

    idx_row = 2
    E = []  # list of the entry objects

    def Commit():

        """
        Read and save the model
        """
        global E

        button_plot["state"] = tkinter.NORMAL
        T = []
        for e in E:
            T.append(e.get())

        with open(file_model, "w") as file:
            file.write("FITTING MODEL INITIALIZATION\n")
            file.write(
                "Function type (column A) : Gaussian, Lorentzian, Bigaussian, Voigt, Pseudo_Voigt\n"
            )
            file.write(
                "B6 offset, B7,B8... C7,C8...must be blank. mask=1/0 the parameter is free/freezed.\
                   Lbd_0, sigma_a, gamma are expressed in cm^-1\n\n"
            )
            file.write(
                " ;offset;mask;Lbd_0;mask;h;mask;sigma_a (w_a);mask;sigma_b;mask;gamma;mask;gamma;mask\n"
            )
            for t in np.array_split(T, len(T) / len(column_names)):
                file.write(";".join(t) + "\n")

    def init_model_(file):

        if file.lower().endswith(".csv"):
            data = pd.read_csv(file, sep=";", header=3, index_col=0)

        elif file.lower().endswith(".xlsx") or file.lower().endswith(".xls"):
            if sheet:
                data = pd.read_excel(file, header=4, index_col=0, sheet_name=sheet)
            else:
                data = pd.read_excel(file, header=4, index_col=0)
        else:
            raise Exception(
                "init_param: file extension not recognize sould be csv, xls or xlsx"
            )

        data.index = data.index + data.groupby(level=0).cumcount().astype(str)
        data.columns = [
            "offset",
            "mask_offset",
            "Lbd_0",
            "mask_Lbd_0",
            "h",
            "mask_h",
            "sigma_a",
            "mask_sigma_a",
            "sigma_b",
            "mask_sigma_b",
            "gamma",
            "mask_gamma",
            "eta",
            "mask_eta",
        ]
        data = data.T.to_dict()
        for function, param in data.items():
            name_function = re.findall("\D*", function)[0]
            default = {
                param_name: param_value
                for param_name, param_value in param.items()
                if not np.isnan(float(param_value))
            }
            set_row(name_function, default)

    def Help():
        pass

    def set_row(name_function, default=None):

        global idx_row, E

        disable = dict_disable[name_function]

        if default is None:
            default = dict_default[name_function]

        param_index = {
            "offset": 1,
            "mask_offset": 2,
            "Lbd_0": 3,
            "mask_Lbd_0": 4,
            "h": 5,
            "mask_h": 6,
            "sigma_a": 7,
            "mask_sigma_a": 8,
            "sigma_b": 9,
            "mask_sigma_b": 10,
            "gamma": 11,
            "mask_gamma": 12,
            "eta": 13,
            "mask_eta": 14,
        }
        default_param = {
            param_index[name_param]: value_param
            for (name_param, value_param) in default.items()
        }

        v = tkinter.StringVar()
        v_int = tkinter.IntVar()
        cell = tkinter.Entry(frame_table, width=10, text=v)
        cell.grid(row=idx_row, column=0)
        v.set(name_function)
        E.append(cell)

        for j in range(1, len(column_names)):
            if ((j == 1 or j == 2) and (idx_row > 2)) or (j in disable):
                cell = tkinter.Entry(frame_table, width=10, state="disable")
                E.append(cell)
            else:
                v_int = tkinter.IntVar()
                cell = tkinter.Entry(frame_table, width=10, text=v_int)
                v_int.set(default_param[j])
                E.append(cell)
            cell.grid(row=idx_row, column=j)
        idx_row += 1

    def plot(noise=None):

        global lbd, z, ax, canvas2, fig

        button_plot["state"] = tkinter.DISABLED
        slide_noise["state"] = tkinter.NORMAL
        button_save["state"] = tkinter.NORMAL

        plot.flag = getattr(plot, "flag", True)

        param_fixed, param_fit, func_type, index_fit, label = init_model(file_model)
        lbd = np.linspace(100, 1400, 1024)
        z = sum_functions(lbd, param_fit, func_type, index_fit, param_fixed)
        z_noise = z

        if plot.flag:
            plot.flag = False
            fig = Figure(figsize=(5, 4), dpi=100)
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(lbd, z)
            ax.set_xlabel("lbd (cm-1)")
            canvas2 = FigureCanvasTkAgg(fig, master=frame_plot1)
            canvas2.draw()
            canvas2.get_tk_widget()
            canvas2.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

            toolbar = NavigationToolbar2Tk(canvas2, frame_plot1)
            toolbar.update()
            canvas2.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        else:
            ax.clear()
            ax.plot(lbd, z_noise)
            canvas2.draw()

    def add_noise(val):

        global z_noise, noise

        noise = slide_noise.get()
        ax.clear()
        z_noise = z + np.random.normal(loc=0.0, scale=noise, size=len(lbd))
        ax.plot(lbd, z_noise)
        canvas2.draw()

    def save_data():

        data = {"lbd": lbd, "intensity": z_noise}
        df = pd.DataFrame.from_dict(data)
        file = r"c:\Temp\DATA" + str(noise) + ".csv"
        df.to_csv(file, sep="\t", index=False)
        tkinter.messagebox.showinfo(
            title=None, message="Your file has been save under: " + file
        )

    # dict of the entry button inndex to disable
    dict_disable = {
        "Gaussian": [9, 10, 11, 12, 13, 14],
        "Lorentzian": [7, 8, 9, 10, 13, 14],
        "Voigt": [9, 10, 13, 14],
        "Pseudo_Voigt": [9, 10],
        "Bigaussian": [11, 12, 13, 14],
    }

    # set default values to the function parameters
    dict_default = {
        "Gaussian": {
            "offset": 0,
            "mask_offset": 0,
            "Lbd_0": 521,
            "mask_Lbd_0": 1,
            "h": 28000,
            "mask_h": 1,
            "sigma_a": 10,
            "mask_sigma_a": 1,
        },
        "Lorentzian": {
            "offset": 0,
            "mask_offset": 0,
            "Lbd_0": 521,
            "mask_Lbd_0": 1,
            "h": 10000,
            "mask_h": 1,
            "gamma": 10,
            "mask_gamma": 1,
        },
        "Voigt": {
            "offset": 0,
            "mask_offset": 0,
            "Lbd_0": 521,
            "mask_Lbd_0": 1,
            "h": 10000,
            "mask_h": 1,
            "sigma_a": 5,
            "mask_sigma_a": 1,
            "gamma": 10,
            "mask_gamma": 1,
        },
        "Pseudo_Voigt": {
            "offset": 0,
            "mask_offset": 0,
            "Lbd_0": 521,
            "mask_Lbd_0": 1,
            "h": 28000,
            "mask_h": 1,
            "sigma_a": 10,
            "mask_sigma_a": 1,
            "gamma": 10,
            "mask_gamma": 1,
            "eta": 0.3,
            "mask_eta": 0,
        },
        "Bigaussian": {
            "offset": 0,
            "mask_offset": 0,
            "Lbd_0": 521,
            "mask_Lbd_0": 1,
            "h": 28000,
            "mask_h": 1,
            "sigma_a": 1.5,
            "mask_sigma_a": 1,
            "sigma_b": 1.5,
            "mask_sigma_b": 1,
        },
    }

    # set the columns name in accordance with the csc file
    column_names = [
        "",
        "offset",
        "mask",
        "Lbd_0",
        "mask",
        "h",
        "mask",
        "sigma_a",
        "mask",
        "sigma_b",
        "mask",
        "gamma",
        "mask",
        "eta",
        "mask",
    ]

    WIDTH_BUTTON = 12
    PADY = 5

    button_Gaussian = tkinter.Button(
        frame_option,
        text="Gaussian",
        width=WIDTH_BUTTON,
        pady=PADY,
        command=lambda: set_row("Gaussian"),
        font=FONTSTYLE,
    )
    button_Gaussian.grid(row=0, column=0)

    button_Lorentzian = tkinter.Button(
        frame_option,
        text="Lorentzian",
        width=WIDTH_BUTTON,
        pady=PADY,
        command=lambda: set_row("Lorentzian"),
        font=FONTSTYLE,
    )
    button_Lorentzian.grid(row=0, column=1)

    button_Voigt = tkinter.Button(
        frame_option,
        text="Voigt",
        width=WIDTH_BUTTON,
        pady=PADY,
        command=lambda: set_row("Voigt"),
        font=FONTSTYLE,
    )
    button_Voigt.grid(row=0, column=2)

    button_Pseudo_Voigt = tkinter.Button(
        frame_option,
        text="Pseudo_Voigt",
        width=WIDTH_BUTTON,
        pady=PADY,
        command=lambda: set_row("Pseudo_Voigt"),
        font=FONTSTYLE,
    )
    button_Pseudo_Voigt.grid(row=0, column=3)

    button_Bigaussian = tkinter.Button(
        frame_option,
        text="Bigaussian",
        width=WIDTH_BUTTON,
        pady=PADY,
        command=lambda: set_row("Bigaussian"),
        font=FONTSTYLE,
    )
    button_Bigaussian.grid(row=0, column=4)

    button_Commit = tkinter.Button(
        frame_option,
        text="Commit",
        width=WIDTH_BUTTON,
        pady=PADY,
        command=Commit,
        font=FONTSTYLE,
    )
    button_Commit.grid(row=0, column=5)

    button_plot = tkinter.Button(
        frame_option,
        text="Plot",
        width=WIDTH_BUTTON,
        pady=PADY,
        command=plot,
        font=FONTSTYLE,
        state=tkinter.DISABLED,
    )
    button_plot.grid(row=0, column=6)

    button_help = tkinter.Button(
        frame_option,
        text="Help",
        width=WIDTH_BUTTON,
        pady=PADY,
        command=Help,
        font=FONTSTYLE,
    )
    button_help.grid(row=0, column=7)

    button_Quit = tkinter.Button(
        frame_option,
        text="Quit",
        width=WIDTH_BUTTON,
        pady=PADY,
        command=fenetre.destroy,
        font=FONTSTYLE,
        foreground="red",
    )
    button_Quit.grid(row=0, column=8)

    for idx_col, column_name in enumerate(column_names):
        column_name = tkinter.Label(frame_table, text=column_name, width=10)
        column_name.grid(row=0, column=idx_col)

    slide_noise = tkinter.Scale(
        frame_action_plot1,
        orient="horizontal",
        from_=0,
        to=1000,
        resolution=1,
        length=150,
        label="noise",
        font=FONTSTYLE,
        command=add_noise,
        state=tkinter.DISABLED,
    )

    slide_noise.grid(row=0, column=0)
    button_save = tkinter.Button(
        frame_action_plot1,
        text="Save data",
        width=WIDTH_BUTTON,
        pady=PADY,
        command=save_data,
        font=FONTSTYLE,
        state=tkinter.DISABLED,
    )
    button_save.grid(row=1, column=0)

    if file_init is not None:
        init_model_(file_init)

    fenetre.mainloop()


def gui_tune_param(da):

    """
    Interactive set up of the Raman parameters (baseline, denoising, peack finding)
    
    Arguments:
        da (xarray) : xarray containing the hyperspectra       
    """

    # Standard Library dependencies
    import tkinter
    from tkinter import messagebox
    import tkinter.font as tkFont
    import itertools

    # 3rd party dependencies
    import numpy as np
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg,
        NavigationToolbar2Tk,
    )
    from matplotlib.figure import Figure
    from matplotlib.backend_bases import key_press_handler
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
    import matplotlib.pyplot as plt
    import pandas as pd
    import pywt
    from scipy.signal import find_peaks
    from pandastable import Table

    # Internal dependencies
    from ..raman_hyperspectra_baseline import flatten_hyperspectra
    from ..raman_hyperspectra_read_files import construct_xarray_0D
    from ..raman_hyperspectra_noise import spectra_noise_estimation_0D
    from ..raman_hyperspectra_denoise import smooth_spectrum

    global da_copy

    def info_noise():

        """
        show the noise std as computed on N_SPLIT slides.

        """
        global da_copy, idx_max_tot, info_peak, std_svg, std_g, lbd_noise_split

        try:
            if window1.state == "normal":
                window1.focus()
        except NameError as e:
            window1 = tkinter.Toplevel()
            window1.geometry("600x600+50+20")
            dic_noise = dict(
                zip(
                    ["lbd_dep", "lbd_end", "std_svg", "std_g"],
                    [
                        [lbd_slice[0] for lbd_slice in lbd_noise_split],
                        [lbd_slice[-1] for lbd_slice in lbd_noise_split],
                        std_svg,
                        std_g,
                    ],
                )
            )
            df = pd.DataFrame.from_dict(dic_noise)
            pt = Table(window1, dataframe=df)
            pt.show()

    def info_max():

        """
        show the peak characteristics : height, prominence, width

        """
        global da_copy, idx_max_tot, info_peak

        LBD_MIN = slide_lbd_min.get()
        LBD_MAX = slide_lbd_max.get()

        try:
            if window1.state == "normal":
                window1.focus()
        except NameError as e:
            window1 = tkinter.Toplevel()
            window1.geometry("600x600+50+20")
            dic = info_peak
            dic["Wavelength"] = da_copy.lbd.values[idx_max_tot]
            df = pd.DataFrame.from_dict(dic)
            df = df.query("Wavelength > @LBD_MIN & Wavelength < @LBD_MAX")
            pt = Table(window1, dataframe=df)
            pt.show()

    def plot():

        global ax, canvas2, fig

        z_max = slide_zoom_z.get()
        l_min = slide_lbd_min.get()
        l_max = slide_lbd_max.get()

        fig = Figure(figsize=(7, 8), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(lbd, z)
        ax.set_ylim(min(z), z_max)
        if l_max > l_min:
            ax.set_xlim(l_min, l_max)
        ax.set_xlabel("lbd (cm-1)")

        canvas2 = FigureCanvasTkAgg(fig, master=frame_plot1)
        canvas2.draw()
        canvas2.get_tk_widget()
        canvas2.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas2, frame_plot1)
        toolbar.update()
        canvas2.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    def action(val):

        global da_copy, idx_max_tot, info_peak, std_svg, std_g, lbd_noise_split

        N_SPLIT = 5  # number of slices for the noise evaluation
        # see spectra_noise_estimation_0D doc
        LBD_DEP = 100  # lower wavelength of the first slice

        button_info_max["state"] = tkinter.NORMAL
        button_info_noise["state"] = tkinter.NORMAL
        type_fiters = type_filter.get()
        z_max = slide_zoom_z.get()
        l_min = slide_lbd_min.get()
        l_max = slide_lbd_max.get()
        idy = slide_idy.get()
        idx = slide_idx.get()
        dc = da_copy.isel(x=idx, y=idy)
        z = dc.values

        db = construct_xarray_0D(z, lbd)
        (
            std_svg_mean,
            std_g_mean,
            std_svg,
            std_g,
            _,
            _,
            lbd_noise_split,
            _,
        ) = spectra_noise_estimation_0D(
            db,
            ("Gauss", 1),
            ("Savitzky-Golay", 13, 6),
            lbd_dep=LBD_DEP,
            n_split=N_SPLIT,
        )

        if type_fiters == "NRWRT":
            wavelet = type_wavelet.get()
            threshold = type_threshold.get()
            ratio = slide_ratio.get()
            decim = slide_decim.get()
            z_smooth = smooth_spectrum(
                z, ("NRWT", decim, wavelet, threshold, ratio), std_svg_mean
            )

        else:
            order = slide_sg_order.get()
            len_SG = 2 * int(slide_sg_var.get() / 2) + 1
            order = min(slide_sg_order.get(), len_SG - 1)
            z_smooth = smooth_spectrum(z, ("Savitzky-Golay", len_SG, order))

        distance_peak = slide_ecart.get()
        prominence_factor = slide_prominence.get()
        height_factor = slide_height.get()
        width_peak = slide_width.get()

        idx_max_tot, info_peak = find_peaks(
            z_smooth,
            distance=distance_peak,
            prominence=prominence_factor * std_svg_mean,
            width=width_peak,
            height=height_factor * std_svg_mean,
        )
        idx_max = list(idx_max_tot)

        ax.clear()
        ax.plot(lbd, z, alpha=0.5)
        ax.plot(lbd, z_smooth)
        ax.plot(lbd[idx_max], z_smooth[idx_max], "or")
        ax.set_title(f"std Gauss:{std_g_mean:.2f},  std svg: {std_svg_mean:.2f}")
        ax.set_ylim(min(z), z_max)
        if l_max > l_min:
            ax.set_xlim(l_min, l_max)
        canvas2.draw()

    def flatten_hyperspectra_():

        global da_copy

        baseline_choice = type_baseline.get()
        da_copy = flatten_hyperspectra(da_copy, LINEBASE[baseline_choice])

    def refresh():

        global da_copy
        da_copy = da.copy()

    fenetre = tkinter.Tk()

    FONTSTYLE = tkFont.Font(family="Lucida Grande", size=10, weight="bold")

    frame_plot1 = tkinter.Frame(fenetre)
    frame_plot1.grid(row=0, column=0)

    frame_action = tkinter.Frame(fenetre)
    frame_action.grid(row=0, column=1)

    type_wavelets = []
    for wt in ["db", "sym", "coif"]:
        type_wavelets.append(pywt.wavelist(wt))
    type_wavelets = list(itertools.chain.from_iterable(type_wavelets))

    TYPE_THRESHOLDS = ["soft", "garrote", "hard"]

    TYPE_FILTERS = ["NRWRT", "SG"]

    LINEBASE = {
        "linear": ("linear", 400, 10),
        "top hat": ("top hat", 440),
        "rubberband": ("rubberband",),
        "drPLS": ("drPLS", 1.0e6, 0.5, 0.001, 100),
        "arPLS": ("arPLS", 1.0e5, 1.0e-3, 40),
        "ials ": ("ials", 10000, 0.01, 0.001),
        "als": ("als", 1.0e5, 1.0e-3, 40),
        "Modpoly": ("Modpoly", 2),
        "Imodpoly": ("Imodpoly", 2),
    }
    TYPE_LINEBASE = list(LINEBASE.keys())

    WIDTH_BUTTON = 12
    PADY = 5
    BD = 5

    da_copy = da.copy()
    dc = da.isel(x=0, y=0)
    lbd = dc.lbd.values
    z = dc.values
    flag_flatten = False

    # frame filter
    frame_filter = tkinter.LabelFrame(frame_action, text="Filter parameters", bd=BD)
    frame_filter.grid(row=0, column=0)

    # frame filter/choose filter
    frame_choose_filter = tkinter.LabelFrame(frame_filter, text="Choose filter")
    frame_choose_filter.grid(row=0, column=0)

    label3 = tkinter.Label(
        frame_choose_filter, text="filter", font=FONTSTYLE, width=WIDTH_BUTTON
    )
    label3.grid(row=0, column=0)
    type_filter = tkinter.StringVar()
    type_filter.set(TYPE_FILTERS[0])
    drop_filter = tkinter.OptionMenu(frame_choose_filter, type_filter, *TYPE_FILTERS)
    drop_filter.config(font=FONTSTYLE)
    drop_filter.grid(row=0, column=1)

    # frame filter/choose linebase
    frame_flatten = tkinter.LabelFrame(frame_filter, text="Flatten")
    frame_flatten.grid(row=0, column=1)
    type_baseline = tkinter.StringVar()
    type_baseline.set(TYPE_LINEBASE[1])
    drop = tkinter.OptionMenu(frame_flatten, type_baseline, *TYPE_LINEBASE)
    drop.config(font=FONTSTYLE)
    drop.grid(row=0, column=0)
    button_flatten = tkinter.Button(
        frame_flatten,
        text="Flatten",
        width=WIDTH_BUTTON,
        pady=PADY,
        command=flatten_hyperspectra_,
        font=FONTSTYLE,
    )
    button_flatten.grid(row=0, column=1)

    # frame filter/choose filter NRWRT
    frame_choose_wavelet = tkinter.LabelFrame(frame_filter, text="Wavelet", pady=21)
    frame_choose_wavelet.grid(row=1, column=0)

    RATIO_DEFAULT = tkinter.DoubleVar()
    RATIO_DEFAULT.set(0.75)
    slide_ratio = tkinter.Scale(
        frame_choose_wavelet,
        orient="horizontal",
        from_=0,
        to=2,
        resolution=0.01,
        length=150,
        label="ratio",
        font=FONTSTYLE,
        command=action,
        variable=RATIO_DEFAULT,
    )
    slide_ratio.grid(row=0, column=0)

    DECIM_DEFAULT = tkinter.DoubleVar()
    DECIM_DEFAULT.set(4)
    slide_decim = tkinter.Scale(
        frame_choose_wavelet,
        orient="horizontal",
        from_=1,
        to=8,
        resolution=1,
        length=150,
        label="decim",
        font=FONTSTYLE,
        command=action,
        variable=DECIM_DEFAULT,
    )
    slide_decim.grid(row=0, column=1)

    label1 = tkinter.Label(
        frame_choose_wavelet, text="wavelet", font=FONTSTYLE, width=WIDTH_BUTTON
    )
    label1.grid(row=1, column=0)
    type_wavelet = tkinter.StringVar()
    type_wavelet.set(type_wavelets[3])
    drop_wavelet = tkinter.OptionMenu(
        frame_choose_wavelet, type_wavelet, *type_wavelets
    )
    drop_wavelet.config(font=FONTSTYLE)
    drop_wavelet.grid(row=2, column=0)

    label2 = tkinter.Label(
        frame_choose_wavelet, text="threshold", font=FONTSTYLE, width=WIDTH_BUTTON
    )
    label2.grid(row=1, column=1)
    type_threshold = tkinter.StringVar()
    type_threshold.set(TYPE_THRESHOLDS[0])
    drop_threshold = tkinter.OptionMenu(
        frame_choose_wavelet, type_threshold, *TYPE_THRESHOLDS
    )
    drop_threshold.config(font=FONTSTYLE)
    drop_threshold.grid(row=2, column=1)

    # frame filter/choose filter svg
    frame_choose_svg = tkinter.LabelFrame(frame_filter, text="svg parameters")
    frame_choose_svg.grid(row=1, column=1)
    SG_LENGTH_DEFAULT = tkinter.DoubleVar()
    SG_LENGTH_DEFAULT.set(9.0)
    slide_sg_var = tkinter.Scale(
        frame_choose_svg,
        orient="vertical",
        from_=3,
        to=22,
        resolution=1,
        length=150,
        label="length SG",
        font=FONTSTYLE,
        command=action,
        variable=SG_LENGTH_DEFAULT,
    )
    slide_sg_var.grid(row=0, column=0)

    SG_ORDER_DEFAULT = tkinter.DoubleVar()
    SG_ORDER_DEFAULT.set(3.0)
    slide_sg_order = tkinter.Scale(
        frame_choose_svg,
        orient="vertical",
        from_=0,
        to=5,
        resolution=1,
        length=150,
        label="Order SG",
        font=FONTSTYLE,
        command=action,
        variable=SG_ORDER_DEFAULT,
    )
    slide_sg_order.grid(row=0, column=1)

    # Frame search max
    frame_max = tkinter.LabelFrame(
        frame_action, text="Peaks search parameters", bd=BD, padx=71
    )
    frame_max.grid(row=1, column=0)

    PROMIN_DEFAULT = tkinter.DoubleVar()
    PROMIN_DEFAULT.set(1.5)
    slide_prominence = tkinter.Scale(
        frame_max,
        orient="vertical",
        from_=0,
        to=10,
        resolution=0.1,
        length=150,
        label="Promin",
        font=FONTSTYLE,
        command=action,
        variable=PROMIN_DEFAULT,
    )
    slide_prominence.grid(row=0, column=0)

    HEIGHT_DEFAULT = tkinter.DoubleVar()
    HEIGHT_DEFAULT.set(1.5)
    slide_height = tkinter.Scale(
        frame_max,
        orient="vertical",
        from_=0,
        to=10,
        resolution=0.1,
        length=150,
        label="Height",
        font=FONTSTYLE,
        command=action,
        variable=HEIGHT_DEFAULT,
    )
    slide_height.grid(row=0, column=1)

    ECART_DEFAULT = tkinter.DoubleVar()
    ECART_DEFAULT.set(3)
    slide_ecart = tkinter.Scale(
        frame_max,
        orient="vertical",
        from_=1,
        to=15,
        resolution=1,
        length=150,
        label="Ecart ",
        font=FONTSTYLE,
        command=action,
        variable=ECART_DEFAULT,
    )
    slide_ecart.grid(row=0, column=2)

    slide_width = tkinter.Scale(
        frame_max,
        orient="vertical",
        from_=1,
        to=15,
        resolution=1,
        length=150,
        label="Width",
        font=FONTSTYLE,
        command=action,
    )
    slide_width.grid(row=0, column=3)

    # Frame plot
    frame_plot = tkinter.LabelFrame(
        frame_action, text="Plot parameters", bd=BD, padx=22
    )
    frame_plot.grid(row=2, column=0)

    Z_MAX_DEFAULT = tkinter.DoubleVar()
    Z_MAX_DEFAULT.set(np.max(da.values))
    slide_zoom_z = tkinter.Scale(
        frame_plot,
        orient="vertical",
        from_=np.max(da.values),
        to=0,
        resolution=5,
        length=150,
        label="zoom",
        font=FONTSTYLE,
        command=action,
        variable=Z_MAX_DEFAULT,
    )
    slide_zoom_z.grid(row=0, column=0)

    LBD_MIN_DEFAULT = tkinter.DoubleVar()
    LBD_MIN_DEFAULT.set(100.0)
    slide_lbd_min = tkinter.Scale(
        frame_plot,
        orient="vertical",
        from_=min(lbd),
        to=max(lbd),
        resolution=5,
        length=150,
        label="l_min",
        font=FONTSTYLE,
        command=action,
        variable=LBD_MIN_DEFAULT,
    )
    slide_lbd_min.grid(row=0, column=1)

    LBD_MAX_DEFAULT = tkinter.DoubleVar()
    LBD_MAX_DEFAULT.set(1200.0)
    slide_lbd_max = tkinter.Scale(
        frame_plot,
        orient="vertical",
        from_=min(lbd),
        to=max(lbd),
        resolution=5,
        length=150,
        label="l_max",
        font=FONTSTYLE,
        command=action,
        variable=LBD_MAX_DEFAULT,
    )
    slide_lbd_max.grid(row=0, column=2)

    slide_idx = tkinter.Scale(
        frame_plot,
        orient="vertical",
        from_=0,
        to=len(da.x) - 1,
        resolution=1,
        length=150,
        label="id_x ",
        font=FONTSTYLE,
        command=action,
    )
    slide_idx.grid(row=0, column=3)

    slide_idy = tkinter.Scale(
        frame_plot,
        orient="vertical",
        from_=0,
        to=len(da.y) - 1,
        resolution=1,
        length=150,
        label="id_y ",
        font=FONTSTYLE,
        command=action,
    )
    slide_idy.grid(row=0, column=4)

    # Frame util
    frame_util = tkinter.LabelFrame(frame_action, text="Util", bd=BD)
    frame_util.grid(row=3, column=0)

    button_info_max = tkinter.Button(
        frame_util,
        text="Info MAX",
        width=WIDTH_BUTTON,
        pady=PADY,
        command=info_max,
        font=FONTSTYLE,
        state=tkinter.DISABLED,
    )
    button_info_max.grid(row=0, column=0)

    button_info_noise = tkinter.Button(
        frame_util,
        text="Info NOISE",
        width=WIDTH_BUTTON,
        pady=PADY,
        command=info_noise,
        font=FONTSTYLE,
        state=tkinter.DISABLED,
    )
    button_info_noise.grid(row=0, column=1)

    button_refresh = tkinter.Button(
        frame_util,
        text="Refresh data",
        width=WIDTH_BUTTON,
        pady=PADY,
        command=refresh,
        font=FONTSTYLE,
    )
    button_refresh.grid(row=0, column=2)

    button_Quit = tkinter.Button(
        frame_util,
        text="Quit",
        width=WIDTH_BUTTON,
        pady=PADY,
        command=fenetre.destroy,
        font=FONTSTYLE,
        foreground="red",
    )
    button_Quit.grid(row=0, column=3)

    plot()
    fenetre.mainloop()
