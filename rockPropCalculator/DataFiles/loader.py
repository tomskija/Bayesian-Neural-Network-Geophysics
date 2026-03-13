"""
Data loading functions for well logs, horizons, and seismic data
"""

import pandas as pd
import numpy as np
from config import DATA_DIR, FILES, EXCEL_SHEETS


def load_well_data(data_dir=None):
    """
    Load well log datasets into Pandas DataFrames
    
    Parameters:
    -----------
    data_dir : str, optional
        Directory containing data files. If None, uses config.DATA_DIR
    
    Returns:
    --------
    tuple : (df_well_lucy, df_well_edwards, df_well_lucy_seismic, df_well_edwards_seismic)
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    filepath = data_dir + FILES['well_data']
    
    df_well_lucy = pd.read_excel(filepath, sheet_name=EXCEL_SHEETS['lucy'])
    df_well_edwards = pd.read_excel(filepath, sheet_name=EXCEL_SHEETS['edwards'])
    df_well_lucy_seismic = pd.read_excel(filepath, sheet_name=EXCEL_SHEETS['lucy_seismic'])
    df_well_edwards_seismic = pd.read_excel(filepath, sheet_name=EXCEL_SHEETS['edwards_seismic'])
    
    print("df_well_lucy length:            ", len(df_well_lucy))
    print("df_well_edwards length:         ", len(df_well_edwards))
    print("df_well_lucy_seismic length:    ", len(df_well_lucy_seismic))
    print("df_well_edwards_seismic length: ", len(df_well_edwards_seismic))
    
    return df_well_lucy, df_well_edwards, df_well_lucy_seismic, df_well_edwards_seismic


def load_horizon_data(data_dir=None):
    """
    Read in horizon data
    
    Parameters:
    -----------
    data_dir : str, optional
        Directory containing data files
    
    Returns:
    --------
    DataFrame : Horizon data
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    filepath = data_dir + FILES['horizons']
    df_horizons = pd.read_excel(filepath, sheet_name=EXCEL_SHEETS['horizon_data'])
    
    print("Horizon data shape:", df_horizons.shape)
    return df_horizons


def load_cleaned_horizon_data(data_dir=None):
    """
    Load pre-processed horizon data with Zp values
    
    Parameters:
    -----------
    data_dir : str, optional
        Directory containing data files
    
    Returns:
    --------
    DataFrame : Cleaned horizon data with Zp
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    filepath = data_dir + FILES['horizons_with_zp']
    df_horizons_updated = pd.read_excel(filepath, sheet_name=EXCEL_SHEETS['horizon_cleaned'])
    df_horizons_updated = df_horizons_updated.drop(columns=['Unnamed: 0'])
    
    print("Cleaned horizon data shape:", df_horizons_updated.shape)
    return df_horizons_updated