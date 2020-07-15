
__all__ = ["renishaw_1d_si", "witek_2d_si"]

def renishaw_1d_si():

    """
    Get an renishaw txt file
    easy use in demos

    """
	
    import os
    from ..raman_hyperspectra_read_files import read_RAMAN_RENISHAW_txt_0D
	
    fname = os.path.join(os.path.dirname(__file__), 'RENISHAW_1D_Si.txt')
    da_sliced, da_sliced_interp, da, da_interp = read_RAMAN_RENISHAW_txt_0D(fname)
    
    return da_sliced, da_sliced_interp, da, da_interp

def witek_2d_si():

    """
    Get an renishaw txt file
    easy use in demos

    """
	
    import os
    from ..raman_hyperspectra_read_files import read_RAMAN_WITEC_2D
	
    fname = os.path.join(os.path.dirname(__file__), 'Large Area Scan_000_Spec_As cut.txt')
    da_sliced, da_sliced_interp, da, da_interp = read_RAMAN_WITEC_2D(fname)
    
    return da_sliced, da_sliced_interp, da, da_interp