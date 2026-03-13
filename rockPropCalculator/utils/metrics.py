"""
Metrics calculation and results reporting
"""

import numpy as np


def calculate_metrics(predictions, targets):
    """
    Calculate comprehensive metrics for model evaluation
    
    Parameters:
    -----------
    predictions : ndarray
        Predicted values
    targets : ndarray
        Target values
    
    Returns:
    --------
    dict : Dictionary of metrics
    """
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # MAPE with safe division
    mape = np.mean(np.abs((targets - predictions) / np.where(targets != 0, targets, 1))) * 100
    
    # Variance explained
    var_explained = 1 - np.var(targets - predictions) / np.var(targets) if np.var(targets) != 0 else 0
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'variance_explained': var_explained
    }


def print_results_summary(results, burn_in_samples=None):
    """
    Print formatted summary of BNN results
    
    Parameters:
    -----------
    results : dict
        Results dictionary from MCMCSampler.sample()
    burn_in_samples : int, optional
        Number of samples used for burn-in (for reporting only)
    """
    print("\n" + "=" * 60)
    print("BAYESIAN NEURAL NETWORK RESULTS")
    print("=" * 60)
    
    if burn_in_samples is not None:
        print(f"Burn-in samples: {burn_in_samples}")
    
    print(f"Acceptance rate: {results['acceptance_rate']:.1f}%")
    print(f"Langevin proposals: {results['n_langevin']}")
    
    print("\n" + "-" * 60)
    print("Training Metrics (Post Burn-in)")
    print("-" * 60)
    
    n_samples = len(results['train_metrics']['rmse'])
    post_burnin = n_samples // 4  # Use last 25% for final metrics
    
    for metric in ['rmse', 'r2', 'mape']:
        mean_val = np.mean(results['train_metrics'][metric][-post_burnin:])
        std_val = np.std(results['train_metrics'][metric][-post_burnin:])
        
        if metric in ['rmse', 'mape']:
            print(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print(f"{metric.upper()}: {mean_val:.4f}")
    
    print("\n" + "-" * 60)
    print("Testing Metrics (Post Burn-in)")
    print("-" * 60)
    
    for metric in ['rmse', 'r2', 'mape']:
        mean_val = np.mean(results['test_metrics'][metric][-post_burnin:])
        std_val = np.std(results['test_metrics'][metric][-post_burnin:])
        
        if metric in ['rmse', 'mape']:
            print(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print(f"{metric.upper()}: {mean_val:.4f}")
    
    print("=" * 60)


def calculate_posterior_statistics(samples):
    """
    Calculate posterior statistics from MCMC samples
    
    Parameters:
    -----------
    samples : ndarray
        MCMC samples (n_samples x dimension)
    
    Returns:
    --------
    dict : Mean, std, median, and credible intervals
    """
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    median = np.median(samples, axis=0)
    
    # 95% credible interval
    lower = np.percentile(samples, 2.5, axis=0)
    upper = np.percentile(samples, 97.5, axis=0)
    
    return {
        'mean': mean,
        'std': std,
        'median': median,
        'ci_lower': lower,
        'ci_upper': upper
    }