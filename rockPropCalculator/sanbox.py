"""
Simple example script showing how to use the modular code
"""

from config import BNN_CONFIG
from models import BayesianNeuralNetwork, MCMCSampler
from utils import print_results_summary, plot_training_results
import numpy as np


def run_simple_example():
    """Run a simple BNN example with synthetic data"""
    
    print("=" * 60)
    print("SIMPLE BNN EXAMPLE")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_train, n_test = 100, 50
    
    # Create non-linear relationship
    x_train = np.random.uniform(0, 1, n_train).reshape(-1, 1)
    y_train = 0.5 * np.sin(4 * np.pi * x_train) + 0.3 * x_train + 0.1 * np.random.randn(n_train, 1)
    train_data = np.hstack([x_train, y_train])
    
    x_test = np.random.uniform(0, 1, n_test).reshape(-1, 1)
    y_test = 0.5 * np.sin(4 * np.pi * x_test) + 0.3 * x_test + 0.1 * np.random.randn(n_test, 1)
    test_data = np.hstack([x_test, y_test])
    
    # Normalize
    train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())
    test_data = (test_data - test_data.min()) / (test_data.max() - test_data.min())
    
    print(f"\nTrain data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Create BNN
    topology = [1, 5, 1]  # Simpler topology for example
    print(f"\nTopology: {topology}")
    
    bnn = BayesianNeuralNetwork(topology, train_data, test_data, learning_rate=0.01)
    sampler = MCMCSampler(bnn, use_langevin=True, langevin_prob=0.5)
    
    # Run sampling
    print("\nRunning MCMC sampling...")
    results = sampler.sample(n_samples=200, burn_in=100, verbose=True)
    
    # Print and plot results
    print_results_summary(results, burn_in=100)
    plot_training_results(results, burn_in_ratio=0.5)
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_simple_example()
