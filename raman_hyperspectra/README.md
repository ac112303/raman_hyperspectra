## Product Name
Toolbox of utilities to read, process and visualize Raman spectra/hyperspectra

## Installation
Run the following to install:
```python
pip install raman_hyperspectra
```
## Methods

### Raman file reading :
 - WITEK 1D/2D .txt files, WITEK information files;
 - RENISHAW 1D/2D .txt/.wdf  files.
### Raman spectra/hyperspectra processing
1. Spikes (cosmics) elimination
	- on hyperspectrum usind PCA or robust PCA
	- on spectra using Outlier detection in large data sets
2. Linebase correction using :
	- Linear baseline removal;
	- Morphological fiter (tophat);
	- Rubberband ;
	- Asymmetric least square;
	- Improved asymmetric least square ;
	- Asymmetrically reweighted penalized least squares smoothing;
	- IModPoly from BaselineRemoval 0.0.3 package
	- Modpoly from BaselineRemoval 0.0.3 package
	- local linebase correction
3. Spectra shift correction using
	- The location of a sharp well referenced peak;
	- The location of the Rayleigh peak.
4. Noise estimaton (std) on spectra/hyperspectra
5. Spectra/hyperspectra denoising using:
	- Savitzky-Golay moving smoothing;
	- Noise reduction by wavelet thresholding.
6. Peaks finding and sorting
7. Phase imaging
8. Other available methods:
	- PCA
	- robust PCA
	- NMF
	- VCA (in test)
9. Interactive graphical user interface for visually browsing of:
	- the spectrum 
	- the phase
	- the package xrviz can provide additional capabilities
<img src="https://github.com/Bertin-fap/raman-hyperspectra-examples/blob/master/animated.gif" width="500" height="500" />

## Usage example
**For more examples and usage, please refer to** [example-raman-hyperspectra](https://github.com/Bertin-fap/raman-hyperspectra-examples/blob/master/raman_hyperspectra%20examples.ipynb).
### read and plot a spectra/hyperspectra
```python
import raman_hyperspectra as rhp

file = "C:\\my_modules_Python\\rhp_demo_files\\RENISHAW_1D_Si.txt"
da_sliced, da_sliced_interp, da, da_interp = rhp.read_RAMAN_RENISHAW_0D(file)
da_sliced_interp.plot()

file = "C:\\my_modules_Python\\rhp_demo_files\\Large Area Scan_000_Spec_As cut.txt"
da_sliced, da_sliced_interp, da_full_spectrum, da_full_spectrum_interp = rhp.read_RAMAN_WITEC_2D(file,200,1000)
da_sliced.sel(x=10,y=20).plot()
```

### linebase correction
```python
z_ials, z_base_ials = rhp.flatten_spectra(da,('ials',10_000,  0.01, 0.001))
z_als,z_base_als = rhp.flatten_spectra(da,('als',10_000, 0.001,20))
z_lin, z_base_lin = rhp.flatten_spectra(da,('linear',500,5))
z_top_hat,z_base_top_hat = rhp.flatten_spectra(da,('top hat',factor))
```

### denoising of a spectrum
```python
z_nrwt = rhp.smooth_spectrum (z, ("Savitzky-Golay",sg_length, sg_order) ) 
z_nrwt = rhp.smooth_spectrum (z, ("NRWT", decNum, wnane, mode, ratio_sigma), std ) 

```

### PCA
```python
hp.Raman_PCA(da_denoise, k_PCA = 6)

```
### NMF
```python
rhp.Raman_NMF(da_denoise, k_NMF = 6)

```


# Release History
0.0.1

# Meta
	- François Bertin– francois.bertin7@wanadoo.fr
	- Amal Chabli- 

Distributed under the [MIT license](https://mit-license.org/)

# About the authors
	- François Bertin now retired was an expert senior in nanocharacterization at CEA-LETI
	- Amal Chabli is director of research in nanocharacterization at CEA-LITEN, will be retired end of october 2020