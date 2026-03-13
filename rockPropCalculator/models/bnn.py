####################################################################################################
"""
Bayesian Neural Network implementation
"""
####################################################################################################
import numpy as np
from models.config import BNN_CONFIG

####################################################################################################
class BayesianNeuralNetwork:
    ################################################################################################
    """Bayesian Neural Network with Langevin dynamics."""
    ################################################################################################
    def __init__(self, topology=None, train_data=None, test_data=None, learning_rate=None):
        """
        Initialize BNN with given topology and data.
        
        Parameters:
        -----------
        topology : list
            [input_size, hidden_size, output_size]
        train_data : ndarray
            Training dataset
        test_data : ndarray
            Testing dataset
        learning_rate : float
            Learning rate for gradient updates
        """
        self.topology   = topology if topology is not None else BNN_CONFIG['topology']
        self.train_data = train_data
        self.test_data  = test_data
        self.lr = learning_rate if learning_rate is not None else BNN_CONFIG['learning_rate']
        
        self._init_weights()
    
    ################################################################################################
    def _init_weights(self):
        """Initialize network weights and biases with Xavier initialization."""
        n_in, n_hid, n_out = self.topology
        self.W1 = np.random.randn(n_in, n_hid) / np.sqrt(n_in)
        self.b1 = np.random.randn(1, n_hid) / np.sqrt(n_hid)
        self.W2 = np.random.randn(n_hid, n_out) / np.sqrt(n_hid)
        self.b2 = np.random.randn(1, n_out) / np.sqrt(n_hid)
    
    ################################################################################################
    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function with numerical stability."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    ################################################################################################
    def forward(self, x):
        """
        Forward pass through network.
        
        Parameters:
        -----------
        x : ndarray
            Input data
        
        Returns:
        --------
        tuple : (hidden_activation, output)
        """
        z1 = x.dot(self.W1) + self.b1
        h1 = self.sigmoid(z1)
        z2 = h1.dot(self.W2) + self.b2
        output = self.sigmoid(z2)
        return h1, output
    
    ################################################################################################
    def backward(self, x, target):
        """
        Backward pass with gradient updates.
        
        Parameters:
        -----------
        x : ndarray
            Input data
        target : ndarray
            Target values
        """
        h1, output = self.forward(x)
        
        # Output layer gradients
        output_error = (target - output) * output * (1 - output)
        self.W2 += self.lr * h1.T.dot(output_error)
        self.b2 += self.lr * np.sum(output_error, axis=0, keepdims=True)
        
        # Hidden layer gradients
        hidden_error = output_error.dot(self.W2.T) * h1 * (1 - h1)
        self.W1 += self.lr * x.T.dot(hidden_error)
        self.b1 += self.lr * np.sum(hidden_error, axis=0, keepdims=True)
    
    ################################################################################################
    def encode_weights(self):
        """
        Flatten all weights and biases into single vector.
        
        Returns:
        --------
        ndarray : Flattened weights
        """
        return np.concatenate([self.W1.ravel(), self.W2.ravel(), self.b1.ravel(), self.b2.ravel()])
    
    ################################################################################################
    def decode_weights(self, w):
        """
        Reshape flattened weights back into network parameters.
        
        Parameters:
        -----------
        w : ndarray
            Flattened weights
        """
        n_in, n_hid, n_out = self.topology
        w1_size = n_in * n_hid
        w2_size = n_hid * n_out
        self.W1 = w[:w1_size].reshape(n_in, n_hid)
        self.W2 = w[w1_size:w1_size + w2_size].reshape(n_hid, n_out)
        self.b1 = w[w1_size + w2_size:w1_size + w2_size + n_hid].reshape(1, n_hid)
        self.b2 = w[w1_size + w2_size + n_hid:].reshape(1, n_out)
    
    ################################################################################################
    def predict(self, data, weights):
        """
        Make predictions using given weights.
        
        Parameters:
        -----------
        data : ndarray
            Input data
        weights : ndarray
            Network weights
        
        Returns:
        --------
        ndarray : Predictions
        """
        self.decode_weights(weights)
        predictions = []
        for i in range(data.shape[0]):
            x = data[i:i + 1, :self.topology[0]]
            _, output = self.forward(x)
            predictions.append(output.flatten())
        return np.array(predictions)
    
    ################################################################################################
    def langevin_gradient(self, data, weights, n_steps=1):
        """
        Apply Langevin gradient updates.
        
        Parameters:
        -----------
        data : ndarray
            Training data
        weights : ndarray
            Current weights
        n_steps : int
            Number of gradient steps
        
        Returns:
        --------
        ndarray : Updated weights
        """
        self.decode_weights(weights)
        for _ in range(n_steps):
            for i in range(data.shape[0]):
                x = data[i:i + 1, :self.topology[0]]
                y = data[i:i + 1, self.topology[0]:]
                self.backward(x, y)
        
        return self.encode_weights()
    
####################################################################################################
