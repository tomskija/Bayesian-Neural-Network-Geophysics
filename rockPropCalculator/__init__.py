"""
Bakken Reservoir Characterization using Bayesian Neural Networks

A modular framework for unconventional reservoir characterization using
Bayesian Neural Networks with Langevin Dynamics MCMC sampling.

Author: Jackson R. Tomski
Institution: University of Texas at Austin - Jackson School of Geosciences
"""

__version__ = "1.0.0"
__author__ = "Jackson R. Tomski"

# Import main components for easy access
from .config import BNN_CONFIG, SEISMIC_CONFIG, LUCY_WELL, EDWARDS_WELL
from .models import BayesianNeuralNetwork, MCMCSampler
from .DataFiles import load_well_data, load_seismic_data
from .utils import print_results_summary, plot_training_results

__all__ = [
    'BNN_CONFIG',
    'SEISMIC_CONFIG',
    'LUCY_WELL',
    'EDWARDS_WELL',
    'BayesianNeuralNetwork',
    'MCMCSampler',
    'load_well_data',
    'load_seismic_data',
    'print_results_summary',
    'plot_training_results'
]