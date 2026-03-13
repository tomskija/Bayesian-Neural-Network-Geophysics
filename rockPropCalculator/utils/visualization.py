"""
Visualization functions for results and diagnostics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from config import PLOT_CONFIG


def plot_training_results(results, burn_in_ratio=0.85, save_path=None):
    """
    Plot training and testing metrics over MCMC iterations
    
    Parameters:
    -----------
    results : dict
        Results from MCMCSampler.sample()
    burn_in_ratio : float
        Ratio of samples used for burn-in
    save_path : str, optional
        Path to save figure
    """
    n_samples = len(results['train_metrics']['rmse'])
    burn_in = int(burn_in_ratio * n_samples)
    
    fig = plt.figure(figsize=(16, 9))
    
    # RMSE
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(results['train_metrics']['rmse'], color='red', label='Training')
    ax1.plot(results['test_metrics']['rmse'], color='green', label='Testing')
    ax1.axvline(x=burn_in, color='black', linestyle='--', alpha=0.5, label='Burn-in')
    ax1.legend()
    ax1.set_title('RMSE Over Iterations', size=16)
    ax1.set_xlabel('Sample Iteration')
    ax1.set_ylabel('RMSE')
    ax1.grid(True, alpha=0.3)
    
    # MAPE
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(results['train_metrics']['mape'], color='red', label='Training')
    ax2.plot(results['test_metrics']['mape'], color='green', label='Testing')
    ax2.axvline(x=burn_in, color='black', linestyle='--', alpha=0.5, label='Burn-in')
    ax2.legend()
    ax2.set_title('MAPE Over Iterations', size=16)
    ax2.set_xlabel('Sample Iteration')
    ax2.set_ylabel('MAPE (%)')
    ax2.grid(True, alpha=0.3)
    
    # R²
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(results['train_metrics']['r2'], color='red', label='Training')
    ax3.plot(results['test_metrics']['r2'], color='green', label='Testing')
    ax3.axvline(x=burn_in, color='black', linestyle='--', alpha=0.5, label='Burn-in')
    ax3.legend()
    ax3.set_title('R² Over Iterations', size=16)
    ax3.set_xlabel('Sample Iteration')
    ax3.set_ylabel('R²')
    ax3.grid(True, alpha=0.3)
    
    # Acceptance rate summary
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    summary_text = f"""
    MCMC Summary
    {'='*40}
    Total Samples: {n_samples}
    Burn-in: {burn_in}
    Acceptance Rate: {results['acceptance_rate']:.1f}%
    Langevin Proposals: {results['n_langevin']}
    
    Final Metrics (Post Burn-in):
    {'-'*40}
    Train RMSE: {np.mean(results['train_metrics']['rmse'][-50:]):.4f}
    Test RMSE: {np.mean(results['test_metrics']['rmse'][-50:]):.4f}
    Train R²: {np.mean(results['train_metrics']['r2'][-50:]):.4f}
    Test R²: {np.mean(results['test_metrics']['r2'][-50:]):.4f}
    """
    ax4.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_predictions_with_uncertainty(x_train, y_train, fx_train_samples,
                                     x_test, y_test, fx_test_samples,
                                     burn_in_ratio=0.85, save_path=None):
    """
    Plot predictions with uncertainty bounds
    
    Parameters:
    -----------
    x_train, y_train : ndarray
        Training data
    fx_train_samples : ndarray
        Training predictions from all samples
    x_test, y_test : ndarray
        Testing data
    fx_test_samples : ndarray
        Testing predictions from all samples
    burn_in_ratio : float
        Ratio for burn-in
    save_path : str, optional
        Path to save figure
    """
    burn_in = int(burn_in_ratio * fx_train_samples.shape[0])
    
    # Calculate statistics
    fx_mu_train = fx_train_samples[burn_in:].mean(axis=0)
    fx_low_train = np.percentile(fx_train_samples[burn_in:], 10, axis=0)
    fx_high_train = np.percentile(fx_train_samples[burn_in:], 90, axis=0)
    
    fx_mu_test = fx_test_samples[burn_in:].mean(axis=0)
    fx_low_test = np.percentile(fx_test_samples[burn_in:], 10, axis=0)
    fx_high_test = np.percentile(fx_test_samples[burn_in:], 90, axis=0)
    
    # Create figure
    fig = plt.figure(figsize=(16, 6))
    
    # Training plot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x_train, y_train, 'o', label='Actual Data', markersize=4)
    ax1.plot(x_train, fx_mu_train, 'r-', label='Predicted Mean', linewidth=2)
    ax1.plot(x_train, fx_low_train, 'g--', label='P10', linewidth=1)
    ax1.plot(x_train, fx_high_train, 'b--', label='P90', linewidth=1)
    ax1.fill_between(x_train.flatten(), fx_low_train, fx_high_train, 
                     facecolor='yellow', alpha=0.3, label='P10-P90 Range')
    ax1.legend(loc='upper right')
    ax1.set_title('Training Data: Predictions with Uncertainty', size=16)
    ax1.set_xlabel('Normalized Input')
    ax1.set_ylabel('Normalized Target')
    ax1.grid(True, alpha=0.3)
    
    # Testing plot
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(x_test, y_test, 'o', label='Actual Data', markersize=4)
    ax2.plot(x_test, fx_mu_test, 'r-', label='Predicted Mean', linewidth=2)
    ax2.plot(x_test, fx_low_test, 'g--', label='P10', linewidth=1)
    ax2.plot(x_test, fx_high_test, 'b--', label='P90', linewidth=1)
    ax2.fill_between(x_test.flatten(), fx_low_test, fx_high_test,
                     facecolor='yellow', alpha=0.3, label='P10-P90 Range')
    ax2.legend(loc='upper right')
    ax2.set_title('Testing Data: Predictions with Uncertainty', size=16)
    ax2.set_xlabel('Normalized Input')
    ax2.set_ylabel('Normalized Target')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_horizon_comparison(df_horizons_bakken, df_horizons_threefork,
                           lucy_well_loc, edwards_well_loc, save_path=None):
    """
    Plot Bakken and Three Forks horizons
    
    Parameters:
    -----------
    df_horizons_bakken : ndarray
        Bakken horizon data (2D)
    df_horizons_threefork : ndarray
        Three Forks horizon data (2D)
    lucy_well_loc : dict
        Lucy well location {'inline': ..., 'xline': ...}
    edwards_well_loc : dict
        Edwards well location
    save_path : str, optional
        Path to save figure
    """
    fig = plt.figure(figsize=(18, 6))
    
    # Calculate difference
    diff = df_horizons_threefork - df_horizons_bakken
    
    # Extent for plots
    shape = df_horizons_bakken.shape
    extent = [1001, 1615, 1001, shape[0] + 1000]
    
    # Bakken horizon
    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(df_horizons_bakken, cmap='RdBu', aspect='auto',
                     extent=extent, origin='lower', interpolation='bicubic')
    ax1.plot(lucy_well_loc['xline'], lucy_well_loc['inline'], 'bo', markersize=10)
    ax1.plot(edwards_well_loc['xline'], edwards_well_loc['inline'], 'go', markersize=10)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.055)
    cbar1.set_label('Time (ms)', rotation=270, labelpad=15)
    ax1.set_title('Bakken Horizon')
    ax1.set_xlabel('Xline')
    ax1.set_ylabel('Inline')
    
    # Three Forks horizon
    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(df_horizons_threefork, cmap='RdBu', aspect='auto',
                     extent=extent, origin='lower', interpolation='bicubic')
    ax2.plot(lucy_well_loc['xline'], lucy_well_loc['inline'], 'bo', markersize=10)
    ax2.plot(edwards_well_loc['xline'], edwards_well_loc['inline'], 'go', markersize=10)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.055)
    cbar2.set_label('Time (ms)', rotation=270, labelpad=15)
    ax2.set_title('Three Forks Horizon')
    ax2.set_xlabel('Xline')
    ax2.set_ylabel('Inline')
    
    # Thickness (difference)
    ax3 = fig.add_subplot(1, 3, 3)
    im3 = ax3.imshow(diff, cmap='viridis', aspect='auto',
                     extent=extent, origin='lower', interpolation='bicubic')
    ax3.plot(lucy_well_loc['xline'], lucy_well_loc['inline'], 'bo', markersize=10)
    ax3.plot(edwards_well_loc['xline'], edwards_well_loc['inline'], 'go', markersize=10)
    cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.055)
    cbar3.set_label('Thickness (ms)', rotation=270, labelpad=15)
    ax3.set_title('Reservoir Thickness')
    ax3.set_xlabel('Xline')
    ax3.set_ylabel('Inline')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_crossplot_analysis(df_well, time_column='TWT_ms', 
                           zp_column='Computed_P_Impedance',
                           tphi_column='TPHI_High_Res',
                           toc_column='TOC',
                           well_name='Lucy', save_path=None):
    """
    Create crossplot analysis for well data
    
    Parameters:
    -----------
    df_well : DataFrame
        Well data
    time_column : str
        Column name for time
    zp_column : str
        Column name for P-Impedance
    tphi_column : str
        Column name for total porosity
    toc_column : str
        Column name for TOC
    well_name : str
        Well identifier for title
    save_path : str, optional
        Path to save figure
    """
    fig = plt.figure(figsize=(18, 6))
    
    # TPHI vs Time
    ax1 = fig.add_subplot(1, 3, 1, label="1")
    ax1.scatter(df_well[time_column], df_well[tphi_column], c='b', alpha=0.6)
    ax1.set_ylabel('Total Porosity (High Resolution)')
    ax1.set_xlabel('Time [ms]')
    ax1.set_title(f'Total Porosity vs Time - {well_name} Well')
    ax1.grid(True, alpha=0.3)
    
    # Zp vs Time (twin axis)
    ax1_twin = fig.add_subplot(1, 3, 1, label="2", frame_on=False)
    ax1_twin.scatter(df_well[time_column], df_well[zp_column], c='g', alpha=0.4)
    ax1_twin.yaxis.tick_right()
    ax1_twin.set_ylabel('P-Impedance ((ft/s)*(g/cc))')
    ax1_twin.yaxis.set_label_position('right')
    ax1_twin.xaxis.tick_top()
    ax1_twin.xaxis.set_label_position('top')
    
    # TPHI vs Zp
    ax2 = fig.add_subplot(1, 3, 2)
    scatter = ax2.scatter(df_well[zp_column], df_well[tphi_column], 
                         c=df_well[time_column], cmap='viridis', alpha=0.6)
    cbar = fig.colorbar(scatter, ax=ax2)
    cbar.set_label('Time (ms)', rotation=270, labelpad=15)
    ax2.set_xlabel('P-Impedance ((ft/s)*(g/cc))')
    ax2.set_ylabel('Total Porosity (High Resolution)')
    ax2.set_title(f'TPHI vs Zp - {well_name} Well')
    ax2.grid(True, alpha=0.3)
    
    # TPHI vs TOC
    ax3 = fig.add_subplot(1, 3, 3)
    scatter = ax3.scatter(df_well[toc_column], df_well[tphi_column],
                         c=df_well[time_column], cmap='viridis', alpha=0.6)
    cbar = fig.colorbar(scatter, ax=ax3)
    cbar.set_label('Time (ms)', rotation=270, labelpad=15)
    ax3.set_xlabel('TOC')
    ax3.set_ylabel('Total Porosity (High Resolution)')
    ax3.set_title(f'TPHI vs TOC - {well_name} Well')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_weight_boxplot(pos_w, save_path=None):
    """
    Create boxplot of posterior weights
    
    Parameters:
    -----------
    pos_w : ndarray
        Posterior weight samples
    save_path : str, optional
        Path to save figure
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot(pos_w)
    ax.set_xlabel('[W1] [B1] [W2] [B2]')
    ax.set_ylabel('Posterior')
    ax.set_title('Boxplot of Posterior Weights and Biases')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_seismic_inline(seismic_data, inline_num, inl_array, xl_array, z,
                       df_horizons=None, df_well_lucy=None, df_well_edwards=None,
                       vmin=None, vmax=None, cb_label='Amplitude', save_path=None):
    """
    Plot a seismic inline with horizons and well locations
    
    Parameters:
    -----------
    seismic_data : ndarray
        3D seismic volume
    inline_num : int
        Inline number to plot
    inl_array : ndarray
        Inline array
    xl_array : ndarray
        Crossline array
    z : ndarray
        Time array
    df_horizons : DataFrame, optional
        Horizon data
    df_well_lucy : DataFrame, optional
        Lucy well data
    df_well_edwards : DataFrame, optional
        Edwards well data
    vmin, vmax : float, optional
        Color scale limits
    cb_label : str
        Colorbar label
    save_path : str, optional
        Path to save figure
    """
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(1, 1, 1)
    
    # Get inline data
    inline_data = seismic_data[:, inl_array.tolist().index(inline_num), :]
    
    # Setup extent
    xleft, xrite = xl_array.min(), xl_array.max()
    twt_min = z[0] + 1600
    twt_max = z[-1] + 1600
    ext = [xleft, xrite, twt_max, twt_min]
    
    # Plot seismic
    im = ax.imshow(inline_data, cmap='RdBu', vmin=vmin, vmax=vmax,
                   aspect='auto', extent=ext, interpolation='bicubic')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.055, shrink=0.75)
    cbar.ax.set_ylabel(cb_label, rotation=270, labelpad=15)
    
    # Plot horizons if provided
    if df_horizons is not None:
        horizon_inline = df_horizons.loc[df_horizons['Inline'].values == inline_num]
        
        if len(horizon_inline) > 0:
            xl_subset = xl_array[inline_data[0] != 0]
            
            bakk_horiz = horizon_inline['SmoothMean01_Bakken_Horizon'].values
            three_fork_horiz = horizon_inline['SmoothMean02_ThreeFork_Horizon'].values
            birdbear_horiz = horizon_inline['SmoothMean03_Birdbear_Horizon'].values
            duperow_horiz = horizon_inline['SmoothMean04_Duperow_Horizon'].values
            
            bakk_horiz = bakk_horiz[inline_data[0] != 0]
            three_fork_horiz = three_fork_horiz[inline_data[0] != 0]
            birdbear_horiz = birdbear_horiz[inline_data[0] != 0]
            duperow_horiz = duperow_horiz[inline_data[0] != 0]
            
            ax.plot(xl_subset, bakk_horiz, color='c', linewidth=2, label='Bakken')
            ax.plot(xl_subset, three_fork_horiz, color='b', linewidth=2, label='Three Forks')
            ax.plot(xl_subset, birdbear_horiz, color='g', linewidth=2, label='Birdbear')
            ax.plot(xl_subset, duperow_horiz, color='r', linewidth=2, label='Duperow')
    
    # Plot wells if at their inline
    from config import LUCY_WELL, EDWARDS_WELL
    
    if inline_num == LUCY_WELL['inline'] and df_well_lucy is not None:
        well_xl = LUCY_WELL['xline']
        well_name_loc = 1593
        apx_max_well_twt = 1828
        
        ax.arrow(well_xl, 0, 0, apx_max_well_twt, width=0.5, color='k',
                length_includes_head=True, head_width=0)
        ax.text(well_xl, well_name_loc, LUCY_WELL['name'],
               ha='center', va='center', rotation='horizontal')
        
        # Plot well tops
        for top_name, md_value in LUCY_WELL['bakken_tops'].items():
            top_data = df_well_lucy.loc[df_well_lucy['MD_KB_ft'] == md_value]
            if len(top_data) > 0:
                twt_value = top_data['TWT_ms'].values[0]
                ax.hlines(y=twt_value, xmin=well_xl - 6, xmax=well_xl + 6,
                         color='black', linewidth=2)
                ax.text(well_xl + 15, twt_value, top_name,
                       ha='center', va='center', rotation='horizontal', color='black')
    
    if inline_num == EDWARDS_WELL['inline'] and df_well_edwards is not None:
        well_xl = EDWARDS_WELL['xline']
        well_name_loc = 1593
        apx_max_well_twt = 1872
        
        ax.arrow(well_xl, 0, 0, apx_max_well_twt, width=0.5, color='k',
                length_includes_head=True, head_width=0)
        ax.text(well_xl, well_name_loc, EDWARDS_WELL['name'],
               ha='center', va='center', rotation='horizontal')
        
        # Plot well tops
        for top_name, md_value in EDWARDS_WELL['bakken_tops'].items():
            top_data = df_well_edwards.loc[df_well_edwards['MD_KB_ft'] == md_value]
            if len(top_data) > 0:
                twt_value = top_data['TWT_ms'].values[0]
                ax.hlines(y=twt_value, xmin=well_xl - 6, xmax=well_xl + 6,
                         color='black', linewidth=2)
                ax.text(well_xl + 15, twt_value, top_name,
                       ha='center', va='center', rotation='horizontal', color='black')
    
    ax.set_title(f'Inline {inline_num}')
    ax.set_xlabel('Xline')
    ax.set_ylabel('Time [ms]')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()