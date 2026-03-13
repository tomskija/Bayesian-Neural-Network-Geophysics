"""
Seismic data loading and visualization functions
"""

import segyio
import numpy as np
import matplotlib.pyplot as plt
from config import SEISMIC_CONFIG


def load_seismic_data(filename):
    """
    Load and parse seismic data from SEGY file
    
    Parameters:
    -----------
    filename : str
        Path to SEGY file
    
    Returns:
    --------
    tuple : (seismic_data, seismic_data_raw, twt, inl_array, xl_array, z, 
             n_traces, twt_n_samples, sample_rate)
    """
    iline_byte = SEISMIC_CONFIG['iline_byte']
    xline_byte = SEISMIC_CONFIG['xline_byte']
    
    with segyio.open(filename, iline=iline_byte, xline=xline_byte) as segyfile:
        segyfile.mmap()
        
        print('Amplitude Inline range:    ' + str(np.amin(segyfile.ilines)) + 
              ' - ' + str(np.amax(segyfile.ilines)))
        print('Amplitude Crossline range: ' + str(np.amin(segyfile.xlines)) + 
              ' - ' + str(np.amax(segyfile.xlines)))
        
        # Get basic attributes
        n_traces = segyfile.tracecount
        sample_rate = segyio.tools.dt(segyfile) / 1000
        twt = segyfile.samples
        twt_n_samples = segyfile.samples.size
        inl = segyfile.ilines
        xl = segyfile.xlines
        
        # Load data
        seismic_data_raw = segyfile.trace.raw[:]
        seismic_data = seismic_data_raw.T
        
        print("seismic_data_raw.shape: ", seismic_data_raw.shape)
        print("seis_data shape: (TWT_ms Samples, Inline * Xline): ", seismic_data.shape)
        
        seismic_data = seismic_data.reshape(twt_n_samples, inl.size, xl.size)
        print("seis_data shape: (TWT_ms Samples, Inline, Xline):  ", seismic_data.shape)
        
        # Create arrays
        inl_min = np.amin(inl)
        inl_max = np.amax(inl)
        xl_min = np.amin(xl)
        xl_max = np.amax(xl)
        inl_array = np.arange(inl_min, inl_max + 1, 1)
        xl_array = np.arange(xl_min, xl_max + 1, 1)
        z = np.arange(0, twt_n_samples * sample_rate, sample_rate)
        
        bin_headers = segyfile.bin
    
    print(f'N Traces: {n_traces}, TWT_N Samples: {twt_n_samples}, Sample rate: {sample_rate}ms')
    
    return seismic_data, seismic_data_raw, twt, inl_array, xl_array, z, n_traces, twt_n_samples, sample_rate


def plot3dseis(seis, d1, d2, d3, d1_sel=None, d2_sel=None, d3_sel=None,
               id1='Z', id2='IL', id3='XL', cmm='RdBu', vmin=None, vmax=None,
               cb_label=None, df_horizons=None, df_well_lucy=None, df_well_edwards=None):
    """
    Plot inline, crossline or horizontal (time/depth) slice from 3D seismic volume.
    
    Parameters:
    -----------
    seis : ndarray
        3D seismic cube, shape (twt.size, inl.size, crl.size)
    d1, d2, d3 : ndarray
        Range of first, second and third dimension
    d1_sel, d2_sel, d3_sel : int, optional
        Select one slice
    id1, id2, id3 : str
        Names for dimensions
    cmm : str
        Colormap
    vmin, vmax : float
        Color scale limits
    cb_label : str
        Colorbar label
    df_horizons : DataFrame, optional
        Horizon data for plotting
    df_well_lucy : DataFrame, optional
        Lucy well data for plotting
    df_well_edwards : DataFrame, optional
        Edwards well data for plotting
    """
    if (d1_sel is None) & (d2_sel is None) & (d3_sel is None):
        d2_sel = d2[int(d2.size / 2)]
    
    if d1_sel is not None:
        ssplot = seis[d1.tolist().index(d1_sel), :, :]
        nome1, nome2, nomey, nomex = id1, str(d1_sel + 1600), id2, id3
        xleft, xrite = d2.min(), d2.max()
        ytop, ybot = d3.min(), d3.max()
    
    if d2_sel is not None:
        ssplot = seis[:, d2.tolist().index(d2_sel), :]
        nome1, nome2, nomex, nomey = id2, str(d2_sel), id3, id1
        xleft, xrite = d3.min(), d3.max()
        ytop, ybot = d1.min(), d1.max()
    
    if d3_sel is not None:
        ssplot = seis[:, :, d3.tolist().index(d3_sel)]
        nome1, nome2, nomex, nomey = id3, str(d3_sel), id2, id1
        xleft, xrite = d2.min(), d2.max()
        ytop, ybot = d1.min(), d1.max()
    
    print('\nInput 3D dimensions: {}'.format(seis.shape))
    print('{0}:   {1} - {2}'.format(id1, d1.min(), d1.max()))
    print('{0}:     {1} - {2}'.format(id2, d2.min(), d2.max()))
    print('{0}:      {1} - {2}'.format(id3, d3.min(), d3.max()))
    print('Dataset to plot dimensions: {}'.format(ssplot.shape))
    
    # Create figure
    if d1_sel is not None:
        fig = plt.figure(figsize=(8, 18))
        ax = fig.add_subplot(1, 1, 1)
        ext = [ytop, ybot, xleft, xrite]
        amp = ax.imshow(ssplot.T.T, cmap=cmm, vmin=vmin, vmax=vmax, aspect='auto',
                       extent=ext, origin='lower', interpolation='bicubic')
        
        # Plot well locations
        from config import LUCY_WELL, EDWARDS_WELL
        ax.plot(LUCY_WELL['xline'], LUCY_WELL['inline'], 'bo', markersize=12)
        ax.plot(EDWARDS_WELL['xline'], EDWARDS_WELL['inline'], 'go', markersize=12)
        
        cbar = fig.colorbar(amp, ax=ax, fraction=0.046, pad=0.055, shrink=0.75)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(cb_label, rotation=270)
    
    # Add grid and labels
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.title(nome1 + ' ' + nome2)
    plt.xlabel(nomex)
    plt.ylabel(nomey)
    plt.show()