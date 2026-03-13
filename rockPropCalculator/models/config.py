####################################################################################################
"""
Configuration file for Bakken Reservoir Characterization
All hyperparameters, paths, and constants in one place
"""
####################################################################################################
# import os
from pathlib import Path
####################################################################################################
# DIRECTORY PATHS - Updated for Docker container
####################################################################################################
# Get the directory where config.py is located
BASE_DIR = Path(__file__).resolve().parent
####################################################################################################
# Data directories
DATA_DIR = BASE_DIR / 'DataFiles'
DATALOAD_DIR = BASE_DIR / 'DataLoad'
RESULTS_DIR = BASE_DIR / 'FiguresAndResults'
####################################################################################################
# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
DATALOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
####################################################################################################
# Convert to strings for compatibility
# DATA_DIR     = str(DATA_DIR) + '/'
# DATALOAD_DIR = str(DATALOAD_DIR) + '/'
# RESULTS_DIR  = str(RESULTS_DIR) + '/'
####################################################################################################
# WELL LOCATIONS
####################################################################################################
LUCY_WELL = {
    'inline': 2108,
    'xline': 1165,
    'name': 'LUCY 11-23H Well',
    'bakken_tops': {
        'BKKNU': 8718.0,  # MD_KB_ft
        'BKKNM': 8733.0,
        'BKKNL': 8775.5,
        'TRFK': 8808.5
    },
    'time_range': (1781.5, 1800.5),  # TWT_ms
    'seismic_time_range': (1781, 1801)
}
EDWARDS_WELL = {
    'inline': 1824,
    'xline': 1073,
    'name': 'Edwards_44_9H Well',
    'bakken_tops': {
        'BKKNU': 9040.0,  # MD_KB_ft
        'BKKNM': 9057.5,
        'BKKNL': 9104.0,
        'TRFK': 9136.0
    },
    'time_range': (1804.0, 1830.5),  # TWT_ms
    'seismic_time_range': (1804, 1831)
}
####################################################################################################
# SEISMIC DATA PARAMETERS
####################################################################################################
SEISMIC_CONFIG = {
    'filename': 'InvertedVol_EdgePreserving_3Samples_Model_20Iter_9Radius_FullVol_Zp_Resampled_1ms.sgy',
    'iline_byte': 189,
    'xline_byte': 193,
    'inline_range': (1001, 2277),
    'xline_range': (1001, 1615),
    'time_range': (1740.0, 1960.0),  # ms
    'vmin': 30000,  # P-Impedance range
    'vmax': 60000
}
####################################################################################################
# BNN HYPERPARAMETERS
####################################################################################################
BNN_CONFIG = {
    'topology': [1, 11, 1],  # [input, hidden, output]
    'learning_rate': 0.01,
    'w_limit': 0.026874,  # Step size for weights
    'tau_limit': 0.0515625,  # Step size for tau
    'l_prob': 2.1625,  # Langevin probability
    'use_langevin': True,
    'n_samples': 400,
    'burn_in_ratio': 0.85,
    'normalization_range': (-1.3875, 1.3875),
    'sigma_squared': 25.0
}

####################################################################################################
# MCMC SAMPLING PARAMETERS
####################################################################################################
MCMC_CONFIG = {
    'w_step': 0.02,
    'tau_step': 0.01,
    'thin': 1,
    'langevin_prob': 0.5
}
####################################################################################################
# DATA PROCESSING
####################################################################################################
FEATURES = {
    'seismic_well': ['Zp_Seismic', 'Tphi_Well'],
    'well_toc': ['Tphi_Well', 'TOC_Well'],
    'combined': [
        'Time_ms', 'X', 'Y', 'P_Velo', 'Zp', 'DPOR_Low_Res', 'DPOR_High_Res',
        'NPHI_Low_Res', 'NPHI_High_Res', 'TPHI_Low_Res', 'TPHI_High_Res',
        'RHOZ', 'TOC'
    ]
}
NULL_VALUE = -999.25
####################################################################################################
# HORIZON INFORMATION
####################################################################################################
HORIZON_CONFIG = {
    'shape': (1277, 615),  # (inline, xline)
    'horizons': [
        'SmoothMean01_Bakken_Horizon',
        'SmoothMean02_ThreeFork_Horizon',
        'SmoothMean03_Birdbear_Horizon',
        'SmoothMean04_Duperow_Horizon'
    ]
}
####################################################################################################
# PLOTTING PARAMETERS
####################################################################################################
PLOT_CONFIG = {
    'colormap': 'RdBu',
    'figure_sizes': {
        'timeslice': (8, 18),
        'inline': (18, 8),
        'xline': (18, 8),
        'results': (16, 9)
    },
    'horizon_colors': {
        'bakken': 'c',
        'threefork': 'b',
        'birdbear': 'g',
        'duperow': 'r'
    },
    'dpi': 300,
    'save_format': 'png'
}
####################################################################################################
# FILE NAMES
####################################################################################################
FILES = {
    'well_data': 'WellLogData.xlsx',
    'well_data_cleaned': 'WellLogData_CleanedUp.xlsx',
    'horizons': 'Bakken_Horizons_3D.xlsx',
    'horizons_with_zp': 'Rounded_Horizon_Data_with_Zp_Data.xlsx',
    'results_tphi': 'Predicted_3D_TPHI_Results.xlsx',
    'results_toc': 'Predicted_3D_TOC_Results.xlsx'
}
####################################################################################################
# SHEETS IN EXCEL FILES
####################################################################################################
EXCEL_SHEETS = {
    'lucy': 'Lucy_New',
    'edwards': 'Edwards_New',
    'lucy_seismic': 'Lucy_Seismic_Scaled_New',
    'edwards_seismic': 'Edwards_Seismic_Scaled_New',
    'lucy_cleaned': 'Lucy_Cleaned',
    'edwards_cleaned': 'Edwards_Cleaned',
    'horizon_data': 'Horizon_Data',
    'horizon_cleaned': 'Horizon_Data_Cleaned'
}
####################################################################################################
# LOGGING CONFIGURATION
####################################################################################################
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_dir': BASE_DIR / 'logs',
    'log_file': 'bakken_bnn.log'
}
# Ensure log directory exists
LOGGING_CONFIG['log_dir'].mkdir(exist_ok=True)
####################################################################################################
