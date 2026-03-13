"""
Data processing and transformation functions
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from config import NULL_VALUE, FEATURES, BNN_CONFIG, HORIZON_CONFIG


def cleanup_well_data(df_well_lucy, df_well_edwards, df_well_lucy_seismic, 
                      df_well_edwards_seismic, data_dir, save=True):
    """
    Cleanup well log data by removing null values
    
    Parameters:
    -----------
    df_well_lucy : DataFrame
        Lucy well log data
    df_well_edwards : DataFrame
        Edwards well log data
    df_well_lucy_seismic : DataFrame
        Lucy seismic-scaled data
    df_well_edwards_seismic : DataFrame
        Edwards seismic-scaled data
    data_dir : str
        Directory to save cleaned data
    save : bool
        Whether to save to Excel file
    
    Returns:
    --------
    tuple : (df_well_lucy, df_well_edwards) - cleaned DataFrames
    """
    # Lucy Well
    df_well_lucy = df_well_lucy.replace(NULL_VALUE, np.NaN)
    df_well_lucy = df_well_lucy.dropna(how='any', axis=0)
    df_well_lucy = df_well_lucy.reset_index(drop=True)
    
    # Edwards Well
    df_well_edwards = df_well_edwards.replace(NULL_VALUE, np.NaN)
    df_well_edwards = df_well_edwards.dropna(how='any', axis=0)
    df_well_edwards = df_well_edwards.reset_index(drop=True)
    
    print("=" * 60)
    print("df_well_lucy length:    ", len(df_well_lucy))
    print("df_well_edwards length: ", len(df_well_edwards))
    print("=" * 60)
    
    if save:
        from config import FILES
        well_filename = data_dir + FILES['well_data_cleaned']
        with pd.ExcelWriter(well_filename) as writer:
            df_well_lucy.to_excel(writer, sheet_name='Lucy_Cleaned')
            df_well_edwards.to_excel(writer, sheet_name='Edwards_Cleaned')
            df_well_lucy_seismic.to_excel(writer, sheet_name='Lucy_Seismic_Scaled_Cleaned')
            df_well_edwards_seismic.to_excel(writer, sheet_name='Edwards_Seismic_Scaled_Cleaned')
        print("Successfully Written to Excel File")
        print("=" * 60)
    
    return df_well_lucy, df_well_edwards


def process_horizons(df_horizons):
    """
    Process and round horizon data for use with seismic
    
    Parameters:
    -----------
    df_horizons : DataFrame
        Raw horizon data
    
    Returns:
    --------
    tuple : Processed horizon DataFrames and arrays
    """
    print("=" * 60)
    
    # Extract horizon columns
    df_horizons_Bakken1 = df_horizons[HORIZON_CONFIG['horizons'][0]].values
    df_horizons_ThreeFork1 = df_horizons[HORIZON_CONFIG['horizons'][1]].values
    df_horizons_Birdbear1 = df_horizons[HORIZON_CONFIG['horizons'][2]].values
    df_horizons_Duperow1 = df_horizons[HORIZON_CONFIG['horizons'][3]].values
    
    print("=" * 60)
    print("df_horizons_Bakken1.shape:     ", df_horizons_Bakken1.shape)
    print("df_horizons_ThreeFork1.shape:  ", df_horizons_ThreeFork1.shape)
    print("df_horizons_Birdbear1.shape:   ", df_horizons_Birdbear1.shape)
    print("df_horizons_Duperow1.shape:    ", df_horizons_Duperow1.shape)
    
    # Reshape to grid
    shape = HORIZON_CONFIG['shape']
    df_horizons_Bakken2 = df_horizons_Bakken1.reshape(shape)
    df_horizons_ThreeFork2 = df_horizons_ThreeFork1.reshape(shape)
    df_horizons_Birdbear2 = df_horizons_Birdbear1.reshape(shape)
    df_horizons_Duperow2 = df_horizons_Duperow1.reshape(shape)
    
    print("=" * 60)
    print("df_horizons_Bakken2.shape:     ", df_horizons_Bakken2.shape)
    print("df_horizons_ThreeFork2.shape:  ", df_horizons_ThreeFork2.shape)
    print("df_horizons_Birdbear2.shape:   ", df_horizons_Birdbear2.shape)
    print("df_horizons_Duperow2.shape:    ", df_horizons_Duperow2.shape)
    print("=" * 60)
    
    # Round for seismic sampling
    df_bakken_horizon_floor = np.floor(df_horizons_Bakken1)
    df_threefork_horizon_ceiling = np.ceil(df_horizons_ThreeFork1)
    
    # Create rounded horizon DataFrame
    bakken_frame = np.array([
        df_horizons['Inline'].values,
        df_horizons['Xline'].values,
        df_horizons['X'].values,
        df_horizons['Y'].values,
        df_bakken_horizon_floor
    ], dtype=np.float64).transpose()
    
    threefork_frame = np.array([
        df_horizons['Inline'].values,
        df_horizons['Xline'].values,
        df_horizons['X'].values,
        df_horizons['Y'].values,
        df_threefork_horizon_ceiling
    ], dtype=np.float64).transpose()
    
    df_bakken_horizon_floor = pd.DataFrame(
        bakken_frame, 
        columns=['Inline', 'Xline', 'X', 'Y', 'Bakken_Horizon_Floor_Time_ms']
    )
    
    df_threefork_horizon_ceiling = pd.DataFrame(
        threefork_frame,
        columns=['Inline', 'Xline', 'X', 'Y', 'ThreeFork_Horizon_Ceiling_Time_ms']
    )
    
    df_bak_tf_horiz_rounded = pd.concat([
        df_bakken_horizon_floor,
        df_threefork_horizon_ceiling['ThreeFork_Horizon_Ceiling_Time_ms']
    ], axis=1)
    
    return (df_bak_tf_horiz_rounded, df_horizons_Bakken2, df_horizons_ThreeFork2,
            df_horizons_Birdbear2, df_horizons_Duperow2)


def feature_selection_and_standardize(df_data1, feature_type='seismic_well'):
    """
    Select features and standardize data for BNN training
    
    Parameters:
    -----------
    df_data1 : DataFrame or ndarray
        Dataset for feature selection
    feature_type : str
        Type of features: 'seismic_well' or 'well_toc'
    
    Returns:
    --------
    DataFrame : Standardized features
    """
    columns = FEATURES[feature_type]
    df_data2 = pd.DataFrame(df_data1, columns=columns)
    
    # Standardize between normalization range
    norm_range = BNN_CONFIG['normalization_range']
    scaler = preprocessing.MinMaxScaler(feature_range=norm_range)
    scaler.fit(df_data2)
    df_data2_normalized = scaler.transform(df_data2)
    df_data2_normalized = pd.DataFrame(df_data2_normalized, columns=columns)
    
    # Keep target feature between 0 and 1 for sigmoid activation
    target_col = columns[1]
    df_data2_normalized[target_col] = (
        (df_data2_normalized[target_col] - df_data2_normalized[target_col].min()) /
        (df_data2_normalized[target_col].max() - df_data2_normalized[target_col].min())
    )
    
    return df_data2_normalized


def prepare_training_testing_data(df_lucy_all_new, df_edwards_all_new, 
                                   train_well='edwards', test_well='lucy'):
    """
    Prepare training and testing datasets from combined well data
    
    Parameters:
    -----------
    df_lucy_all_new : DataFrame
        Combined Lucy well data
    df_edwards_all_new : DataFrame
        Combined Edwards well data
    train_well : str
        Which well to use for training ('lucy' or 'edwards')
    test_well : str
        Which well to use for testing ('lucy' or 'edwards')
    
    Returns:
    --------
    tuple : (traindata, testdata) as numpy arrays
    """
    # Select appropriate wells
    if train_well == 'edwards':
        df_train = df_edwards_all_new.iloc[:, [4, 9]]
        df_test = df_lucy_all_new.iloc[:, [4, 9]]
    else:
        df_train = df_lucy_all_new.iloc[:, [4, 9]]
        df_test = df_edwards_all_new.iloc[:, [4, 9]]
    
    training_data = df_train.iloc[:, [0, 1]].values
    testing_data = df_test.iloc[:, [0, 1]].values
    
    training_data = feature_selection_and_standardize(training_data, 'seismic_well')
    testing_data = feature_selection_and_standardize(testing_data, 'seismic_well')
    
    training_data = training_data.iloc[:, [0, 1]]
    testing_data = testing_data.iloc[:, [0, 1]]
    
    print("=" * 60)
    print("Training Data Shape: ", training_data.shape)
    print("Testing Data Shape:  ", testing_data.shape)
    print("=" * 60)
    
    traindata = np.asarray(training_data.to_numpy())
    testdata = np.asarray(testing_data.to_numpy())
    
    return traindata, testdata


def create_combined_well_data(df_well, df_well_seismic, df_seismic_zp, time_range, seismic_time_range):
    """
    Create combined well and seismic data for a single well
    
    Parameters:
    -----------
    df_well : DataFrame
        Well log data
    df_well_seismic : DataFrame
        Seismic-scaled well data
    df_seismic_zp : DataFrame
        Seismic Zp data at well location
    time_range : tuple
        (min_time, max_time) for well data in TWT_ms
    seismic_time_range : tuple
        (min_time, max_time) for seismic data in Time_ms
    
    Returns:
    --------
    DataFrame : Combined well and seismic data
    """
    # Filter seismic data
    df_well_seismic_subset = df_well_seismic[
        (df_well_seismic['Time_ms'] >= seismic_time_range[0]) &
        (df_well_seismic['Time_ms'] <= seismic_time_range[1])
    ]
    df_well_seismic_subset = df_well_seismic_subset.reset_index(drop=True)
    
    # Filter well data
    df_well_subset = df_well[
        (df_well['TWT_ms'] >= time_range[0]) &
        (df_well['TWT_ms'] <= time_range[1])
    ]
    df_well_subset = df_well_subset.drop(columns=['MD_KB_ft'])
    df_well_subset = df_well_subset.reset_index(drop=True)
    
    # Combine data
    np_seismic_time = df_seismic_zp['Time_ms'].values
    np_x = df_well_seismic_subset['X'].values
    np_y = df_well_seismic_subset['Y'].values
    np_p_velo = df_well_seismic_subset['P_Velo'].values
    np_zp = df_seismic_zp['Zp'].values
    np_dpor_low = df_well_seismic_subset['DPOR_Low_Res'].values
    np_dpor_high = df_well_seismic_subset['DPOR_High_Res'].values
    np_nphi_low = df_well_seismic_subset['NPHI_Low_Res'].values
    np_nphi_high = df_well_seismic_subset['NPHI_High_Res'].values
    np_tphi_low = df_well_seismic_subset['TPHI_Low_Res'].values
    np_tphi_high = df_well_seismic_subset['TPHI_High_Res'].values
    np_rhoz = df_well_seismic_subset['RHOZ'].values
    np_toc = df_well_seismic_subset['TOC'].values
    
    np_all_new = np.vstack([
        np_seismic_time, np_x, np_y, np_p_velo, np_zp,
        np_dpor_low, np_dpor_high, np_nphi_low, np_nphi_high,
        np_tphi_low, np_tphi_high, np_rhoz, np_toc
    ]).transpose()
    
    df_all_new = pd.DataFrame(np_all_new, columns=[
        'Time_ms', 'X', 'Y', 'P_Velo_from_Well', 'Zp_from_Seismic',
        'DPOR_Low_Res_from_Well', 'DPOR_High_Res_from_Well',
        'NPHI_Low_Res_from_Well', 'NPHI_High_Res_from_Well',
        'TPHI_Low_Res_from_Well', 'TPHI_High_Res_from_Well', 'RHOZ', 'TOC'
    ])
    
    # Stack with well subset
    np_well_subset = df_well_subset.values
    np_all = np.vstack([df_all_new.values, np_well_subset])
    
    df_all = pd.DataFrame(np_all, columns=FEATURES['combined'])
    df_all = df_all.sort_values('Time_ms', ascending=True)
    df_all = df_all.reset_index(drop=True)
    
    return df_all