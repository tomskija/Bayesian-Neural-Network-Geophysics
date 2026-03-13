"""
Unit tests for Bayesian Neural Network
"""

import pytest
import numpy as np
from rockPropCalculator.models import BayesianNeuralNetwork, MCMCSampler


def test_bnn_initialization():
    """Test BNN initializes correctly"""
    topology = [1, 5, 1]
    train_data = np.random.randn(10, 2)
    test_data = np.random.randn(5, 2)
    
    bnn = BayesianNeuralNetwork(topology, train_data, test_data)
    
    assert bnn.topology == topology
    assert bnn.W1.shape == (1, 5)
    assert bnn.W2.shape == (5, 1)


def test_forward_pass():
    """Test forward pass produces correct output shape"""
    topology = [1, 5, 1]
    train_data = np.random.randn(10, 2)
    test_data = np.random.randn(5, 2)
    
    bnn = BayesianNeuralNetwork(topology, train_data, test_data)
    
    x = np.random.randn(1, 1)
    h1, output = bnn.forward(x)
    
    assert h1.shape == (1, 5)
    assert output.shape == (1, 1)
    assert 0 <= output[0, 0] <= 1  # Sigmoid output


def test_encode_decode_weights():
    """Test weight encoding and decoding"""
    topology = [1, 5, 1]
    train_data = np.random.randn(10, 2)
    test_data = np.random.randn(5, 2)
    
    bnn = BayesianNeuralNetwork(topology, train_data, test_data)
    
    # Save original weights
    W1_orig = bnn.W1.copy()
    W2_orig = bnn.W2.copy()
    
    # Encode and decode
    w = bnn.encode_weights()
    bnn.decode_weights(w)
    
    # Check they match
    assert np.allclose(bnn.W1, W1_orig)
    assert np.allclose(bnn.W2, W2_orig)


def test_mcmc_sampler():
    """Test MCMC sampler runs without errors"""
    topology = [1, 3, 1]
    train_data = np.random.randn(20, 2)
    test_data = np.random.randn(10, 2)
    
    # Normalize
    train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())
    test_data = (test_data - test_data.min()) / (test_data.max() - test_data.min())
    
    bnn = BayesianNeuralNetwork(topology, train_data, test_data)
    sampler = MCMCSampler(bnn)
    
    # Run short sampling
    results = sampler.sample(n_samples=10, burn_in=5, verbose=False)
    
    assert 'weight_samples' in results
    assert 'tau_samples' in results
    assert results['weight_samples'].shape[0] == 10
    assert 0 <= results['acceptance_rate'] <= 100
