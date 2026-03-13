####################################################################################################
"""
MCMC sampler for Bayesian Neural Network
"""
####################################################################################################
import numpy as np
import math
import random
from models.config import BNN_CONFIG, MCMC_CONFIG

####################################################################################################
class MCMCSampler:
    """MCMC sampler for Bayesian Neural Network."""
    
    def __init__(self, bnn, use_langevin=None, langevin_prob=None):
        """
        Initialize MCMC sampler.
        
        Parameters:
        -----------
        bnn : BayesianNeuralNetwork
            BNN instance
        use_langevin : bool, optional
            Whether to use Langevin dynamics
        langevin_prob : float, optional
            Probability of using Langevin proposal
        """
        self.bnn = bnn
        self.use_langevin = use_langevin if use_langevin is not None else BNN_CONFIG['use_langevin']
        self.langevin_prob = langevin_prob if langevin_prob is not None else MCMC_CONFIG['langevin_prob']
    
    @staticmethod
    def compute_metrics(predictions, targets):
        """
        Compute evaluation metrics.
        
        Parameters:
        -----------
        predictions : ndarray
            Predicted values
        targets : ndarray
            Target values
        
        Returns:
        --------
        dict : Metrics (mse, rmse, r2, mape)
        """
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        
        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # MAPE with safe division
        mape = np.mean(np.abs((targets - predictions) / np.where(targets != 0, targets, 1))) * 100
        
        return {'mse': mse, 'rmse': rmse, 'r2': r2, 'mape': mape}
    
    def log_likelihood(self, data, weights, tau_sq):
        """
        Compute log likelihood of data given weights.
        
        Parameters:
        -----------
        data : ndarray
            Dataset
        weights : ndarray
            Network weights
        tau_sq : float
            Noise variance
        
        Returns:
        --------
        tuple : (log_likelihood, predictions)
        """
        predictions = self.bnn.predict(data, weights)
        targets = data[:, self.bnn.topology[0]:]
        
        # Flatten if needed
        if predictions.ndim > 1 and predictions.shape[1] == 1:
            predictions = predictions.flatten()
        if targets.ndim > 1 and targets.shape[1] == 1:
            targets = targets.flatten()
        
        residuals = targets - predictions
        log_lik = (-0.5 * np.sum(residuals ** 2) / tau_sq - 
                   0.5 * len(residuals) * np.log(2 * math.pi * tau_sq))
        
        return log_lik, predictions
    
    def log_prior(self, weights, tau_sq, sigma_sq=None):
        """
        Compute log prior of weights and tau.
        
        Parameters:
        -----------
        weights : ndarray
            Network weights
        tau_sq : float
            Noise variance
        sigma_sq : float, optional
            Prior variance for weights
        
        Returns:
        --------
        float : Log prior
        """
        if sigma_sq is None:
            sigma_sq = BNN_CONFIG['sigma_squared']
        
        w_prior = (-0.5 * np.sum(weights ** 2) / sigma_sq - 
                   0.5 * len(weights) * np.log(2 * math.pi * sigma_sq))
        tau_prior = -np.log(tau_sq)  # Jeffreys prior
        
        return w_prior + tau_prior
    
    def sample(self, n_samples=None, w_step=None, tau_step=None,
               burn_in=None, thin=None, verbose=False):
        """
        Run MCMC sampling.
        
        Parameters:
        -----------
        n_samples : int, optional
            Number of samples to draw
        w_step : float, optional
            Step size for weight proposals
        tau_step : float, optional
            Step size for tau proposals
        burn_in : int, optional
            Number of burn-in samples
        thin : int, optional
            Thinning interval
        verbose : bool
            Whether to print progress
        
        Returns:
        --------
        dict : Dictionary containing samples and diagnostics
        """
        # Set defaults from config
        if n_samples is None:
            n_samples = BNN_CONFIG['n_samples']
        if w_step is None:
            w_step = MCMC_CONFIG['w_step']
        if tau_step is None:
            tau_step = MCMC_CONFIG['tau_step']
        if burn_in is None:
            burn_in = int(BNN_CONFIG['burn_in_ratio'] * n_samples)
        if thin is None:
            thin = MCMC_CONFIG['thin']
        
        # Calculate network size
        n_weights = (self.bnn.topology[0] * self.bnn.topology[1] +
                    self.bnn.topology[1] * self.bnn.topology[2] +
                    self.bnn.topology[1] + self.bnn.topology[2])
        
        # Storage
        total_samples = burn_in + n_samples * thin
        weight_samples = np.zeros((n_samples, n_weights))
        tau_samples = np.zeros(n_samples)
        train_metrics = {k: np.zeros(n_samples) for k in ['rmse', 'r2', 'mape']}
        test_metrics = {k: np.zeros(n_samples) for k in ['rmse', 'r2', 'mape']}
        
        # Initialize chain
        weights = np.random.randn(n_weights) * 0.1
        log_tau = 0.0
        tau_sq = np.exp(log_tau)
        
        # Current state
        train_log_lik, train_pred = self.log_likelihood(self.bnn.train_data, weights, tau_sq)
        test_log_lik, test_pred = self.log_likelihood(self.bnn.test_data, weights, tau_sq)
        current_log_prior = self.log_prior(weights, tau_sq)
        current_log_post = train_log_lik + current_log_prior
        
        n_accepted = 0
        n_langevin = 0
        sample_idx = 0
        
        for i in range(total_samples):
            # Propose new weights
            if self.use_langevin and np.random.rand() < self.langevin_prob:
                # Langevin proposal
                grad_weights = self.bnn.langevin_gradient(self.bnn.train_data, weights.copy())
                w_proposal = np.random.normal(grad_weights, w_step)
                n_langevin += 1
                
                grad_weights_prop = self.bnn.langevin_gradient(self.bnn.train_data, w_proposal.copy())
                log_prop_ratio = -0.5 * (np.sum((weights - grad_weights_prop) ** 2) -
                                        np.sum((w_proposal - grad_weights) ** 2)) / w_step ** 2
            else:
                # Random walk proposal
                w_proposal = weights + np.random.normal(0, w_step, n_weights)
                log_prop_ratio = 0.0
            
            # Propose new tau
            log_tau_proposal = log_tau + np.random.normal(0, tau_step)
            tau_sq_proposal = np.exp(log_tau_proposal)
            
            # Compute proposal
            train_log_lik_prop, train_pred_prop = self.log_likelihood(
                self.bnn.train_data, w_proposal, tau_sq_proposal)
            test_log_lik_prop, test_pred_prop = self.log_likelihood(
                self.bnn.test_data, w_proposal, tau_sq_proposal)
            prop_log_prior = self.log_prior(w_proposal, tau_sq_proposal)
            prop_log_post = train_log_lik_prop + prop_log_prior
            
            # Metropolis-Hastings
            log_alpha = min(0, prop_log_post - current_log_post + log_prop_ratio)
            
            if np.log(np.random.rand()) < log_alpha:
                # Accept
                weights = w_proposal
                log_tau = log_tau_proposal
                tau_sq = tau_sq_proposal
                current_log_post = prop_log_post
                train_pred = train_pred_prop
                test_pred = test_pred_prop
                n_accepted += 1
            
            # Store samples after burn-in
            if i >= burn_in and (i - burn_in) % thin == 0:
                weight_samples[sample_idx] = weights
                tau_samples[sample_idx] = tau_sq
                
                # Compute metrics
                train_targets = self.bnn.train_data[:, self.bnn.topology[0]:].flatten()
                test_targets = self.bnn.test_data[:, self.bnn.topology[0]:].flatten()
                
                train_metrics_curr = self.compute_metrics(train_pred, train_targets)
                test_metrics_curr = self.compute_metrics(test_pred, test_targets)
                
                for key in train_metrics:
                    train_metrics[key][sample_idx] = train_metrics_curr[key]
                    test_metrics[key][sample_idx] = test_metrics_curr[key]
                
                sample_idx += 1
                
                if verbose and sample_idx % max(1, n_samples // 10) == 0:
                    print(f"Sample {sample_idx}/{n_samples}, "
                          f"Train RMSE: {train_metrics_curr['rmse']:.4f}, "
                          f"Test RMSE: {test_metrics_curr['rmse']:.4f}")
        
        acceptance_rate = n_accepted / total_samples * 100
        
        if verbose:
            print(f"\nSampling complete!")
            print(f"Acceptance rate: {acceptance_rate:.1f}%")
            print(f"Langevin proposals: {n_langevin}")
        
        return {
            'weight_samples': weight_samples,
            'tau_samples': tau_samples,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'acceptance_rate': acceptance_rate,
            'n_langevin': n_langevin
        }