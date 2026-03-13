"""
Main execution script for Bakken Reservoir Characterization
Using Bayesian Neural Networks with Langevin Dynamics

Author: Jackson R. Tomski
Institution: University of Texas at Austin - Jackson School of Geosciences
"""

import os
import time
import numpy as np
import warnings
from config import (DATA_DIR, RESULTS_DIR, SEISMIC_CONFIG, BNN_CONFIG, 
                   LUCY_WELL, EDWARDS_WELL)
from DataFiles import (load_well_data, load_horizon_data, cleanup_well_data,
                  process_horizons, prepare_training_testing_data, 
                  load_seismic_data)
from models import BayesianNeuralNetwork, MCMCSampler
from utils import (print_results_summary, plot_training_results, 
                   plot_predictions_with_uncertainty, plot_horizon_comparison)

warnings.filterwarnings('ignore')


def setup_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Directories initialized")


def part1_data_loading_and_cleaning():
    """
    Part 1: Load and clean well log, horizon, and seismic data
    
    Returns:
    --------
    dict : Dictionary containing all loaded and cleaned data
    """
    print("\n" + "=" * 80)
    print("PART 1: DATA LOADING AND CLEANING")
    print("=" * 80)
    
    print("\nLoading well data...")
    df_well_lucy, df_well_edwards, df_well_lucy_seismic, df_well_edwards_seismic = load_well_data()
    
    print("\nCleaning well data...")
    df_well_lucy, df_well_edwards = cleanup_well_data(
        df_well_lucy, df_well_edwards,
        df_well_lucy_seismic, df_well_edwards_seismic,
        DATA_DIR, save=True
    )
    
    print("\nLoading horizon data...")
    df_horizons = load_horizon_data()
    
    print("\nProcessing horizons...")
    (df_bak_tf_horiz_rounded, df_horizons_Bakken2,
     df_horizons_ThreeFork2, df_horizons_Birdbear2,
     df_horizons_Duperow2) = process_horizons(df_horizons)
    
    print("\nLoading seismic data...")
    seismic_filename = os.path.join(DATA_DIR, SEISMIC_CONFIG['filename'])
    (seismic_data, seismic_data_raw, twt, inl_array, xl_array,
     z, n_traces, twt_n_samples, sample_rate) = load_seismic_data(seismic_filename)
    
    return {
        'well_lucy': df_well_lucy,
        'well_edwards': df_well_edwards,
        'well_lucy_seismic': df_well_lucy_seismic,
        'well_edwards_seismic': df_well_edwards_seismic,
        'horizons': df_horizons,
        'horizons_rounded': df_bak_tf_horiz_rounded,
        'horizons_bakken': df_horizons_Bakken2,
        'horizons_threefork': df_horizons_ThreeFork2,
        'seismic_data': seismic_data,
        'seismic_data_raw': seismic_data_raw,
        'twt': twt,
        'inl_array': inl_array,
        'xl_array': xl_array,
        'z': z,
        'n_traces': n_traces,
        'twt_n_samples': twt_n_samples,
        'sample_rate': sample_rate
    }


def part2_cluster_analysis(data_dict):
    """
    Part 2: Perform cluster analysis to understand data relationships
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing loaded data from Part 1
    
    Returns:
    --------
    dict : Combined well datasets for further analysis
    """
    print("\n" + "=" * 80)
    print("PART 2: CLUSTER ANALYSIS")
    print("=" * 80)
    
    # This would contain your cluster analysis code
    # For now, returning placeholder
    print("\nNote: Cluster analysis code to be implemented")
    print("This section analyzes relationships between:")
    print("  - Seismic attributes")
    print("  - Well log properties")
    print("  - Petrophysical targets")
    
    return {
        'lucy_combined': None,  # Would be df_lucy_all_new
        'edwards_combined': None  # Would be df_edwards_all_new
    }


def part3_prepare_train_test_data(cluster_results):
    """
    Part 3: Prepare training and testing datasets
    
    Parameters:
    -----------
    cluster_results : dict
        Results from cluster analysis
    
    Returns:
    --------
    dict : Training and testing datasets
    """
    print("\n" + "=" * 80)
    print("PART 3: PREPARING TRAINING AND TESTING DATA")
    print("=" * 80)
    
    # For demonstration - replace with actual combined data
    print("\nNote: Using example data structure")
    print("In production, replace with actual combined well data from Part 2")
    
    # Create dummy data for demonstration
    n_train, n_test = 100, 50
    train_data = np.random.randn(n_train, 2)
    test_data = np.random.randn(n_test, 2)
    
    # Normalize
    train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())
    test_data = (test_data - test_data.min()) / (test_data.max() - test_data.min())
    
    print(f"\nTraining data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    
    return {
        'train_data': train_data,
        'test_data': test_data
    }


def part4_train_bnn(train_test_data):
    """
    Part 4: Train Bayesian Neural Network with hyperparameter tuning
    
    Parameters:
    -----------
    train_test_data : dict
        Training and testing datasets
    
    Returns:
    --------
    dict : BNN results including samples and metrics
    """
    print("\n" + "=" * 80)
    print("PART 4: BAYESIAN NEURAL NETWORK TRAINING")
    print("=" * 80)
    
    # Extract data
    train_data = train_test_data['train_data']
    test_data = train_test_data['test_data']
    
    # Setup BNN
    topology = BNN_CONFIG['topology']
    learning_rate = BNN_CONFIG['learning_rate']
    
    print(f"\nBNN Configuration:")
    print(f"  Topology: {topology}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight step size: {BNN_CONFIG['w_limit']}")
    print(f"  Tau step size: {BNN_CONFIG['tau_limit']}")
    print(f"  Langevin probability: {BNN_CONFIG['l_prob']}")
    
    # Create BNN
    bnn = BayesianNeuralNetwork(topology, train_data, test_data, learning_rate)
    sampler = MCMCSampler(bnn, use_langevin=BNN_CONFIG['use_langevin'])
    
    # Run sampling
    print("\nRunning MCMC sampling...")
    n_samples = BNN_CONFIG['n_samples']
    burn_in = int(BNN_CONFIG['burn_in_ratio'] * n_samples)
    
    timer_start = time.time()
    
    results = sampler.sample(
        n_samples=n_samples,
        w_step=BNN_CONFIG['w_limit'],
        tau_step=BNN_CONFIG['tau_limit'],
        burn_in=burn_in,
        verbose=True
    )
    
    timer_end = time.time()
    execution_time = timer_end - timer_start
    
    print(f"\nSampling completed in {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    # Add execution time to results
    results['execution_time'] = execution_time
    results['burn_in'] = burn_in
    
    return results


def part5_analyze_and_visualize(results, train_test_data):
    """
    Part 5: Analyze results and create visualizations
    
    Parameters:
    -----------
    results : dict
        BNN results from Part 4
    train_test_data : dict
        Training and testing data
    """
    print("\n" + "=" * 80)
    print("PART 5: RESULTS ANALYSIS AND VISUALIZATION")
    print("=" * 80)
    
    # Print summary
    print_results_summary(results, results['burn_in'])
    
    # Plot training metrics
    print("\nGenerating training metrics plot...")
    plot_training_results(
        results,
        BNN_CONFIG['burn_in_ratio'],
        save_path=os.path.join(RESULTS_DIR, 'training_metrics.png')
    )
    
    # Plot weight boxplot
    print("\nGenerating weight distribution plot...")
    from utils.visualization import plot_weight_boxplot
    plot_weight_boxplot(
        results['weight_samples'],
        save_path=os.path.join(RESULTS_DIR, 'weight_posterior.png')
    )
    
    print("\nAll visualizations saved to:", RESULTS_DIR)


def part6_3d_prediction(data_dict, results):
    """
    Part 6: Predict petrophysical properties across 3D seismic volume
    
    Parameters:
    -----------
    data_dict : dict
        All loaded data
    results : dict
        Trained BNN results
    """
    print("\n" + "=" * 80)
    print("PART 6: 3D PETROPHYSICAL PREDICTION")
    print("=" * 80)
    
    print("\nNote: 3D prediction is computationally intensive")
    print("This section would predict TPHI and TOC across the entire seismic volume")
    print("Using the trained BNN from Part 4")
    
    # This would contain your 3D