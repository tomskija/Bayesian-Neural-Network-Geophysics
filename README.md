# Bakken Reservoir Characterization using Bayesian Neural Networks

Master's Thesis Code by Jackson R. Tomski  
University of Texas at Austin - Jackson School of Geosciences - Institute for Geophysics

## Overview

This project performs unconventional reservoir characterization on Bakken field data using Bayesian Neural Networks. Two implementations are provided:

1. **NumPy Implementation** (`models/`) - Pure Python with NumPy, educational
2. **TensorFlow Implementation** (`models_tf/`) - GPU-accelerated, production-ready

## Quick Start with Docker

### Prerequisites
- Docker Desktop installed
- VS Code with Remote-Containers extension (optional)

### Option 1: VS Code Dev Container (Recommended)

1. Open the project folder in VS Code
2. Press `F1` and select "Dev Containers: Reopen in Container"
3. Wait for container to build (first time only)
4. Open terminal in VS Code and run:
```bash
cd rockPropCalculator
python main_tf.py
```

### Option 2: Docker Compose
```bash
# Build and start container
docker-compose up -d

# Enter container
docker exec -it bakken-bnn-dev bash

# Run workflow
cd rockPropCalculator
python main_tf.py
```

### Option 3: Direct Docker
```bash
# Build image
docker build -t bakken-bnn -f .devcontainer/Dockerfile .

# Run container
docker run -it -v $(pwd):/workspace bakken-bnn

# Inside container
cd rockPropCalculator
python main_tf.py
```

## Installation (Local/Non-Docker)

### Basic Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### GPU Support (Optional)

For NVIDIA GPU acceleration:
```bash
# Install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]>=2.12.0

# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Project Structure
```
rockPropCalculator/
├── config.py              # Configuration and constants
├── main.py               # NumPy workflow
├── main_tf.py            # TensorFlow workflow
├── run_example.py        # NumPy example
├── run_example_tf.py     # TensorFlow example
├── Data/                 # Data files
├── FiguresAndResults/    # Output directory
├── data/                 # Data processing modules
│   ├── loader.py
│   ├── processor.py
│   └── seismic.py
├── models/              # NumPy BNN implementation
│   ├── bnn.py
│   └── mcmc.py
├── models_tf/           # TensorFlow BNN implementation
│   ├── bnn_tf.py
│   ├── mcmc_tf.py
│   ├── layers_tf.py
│   └── config_tf.py
├── utils/               # Utilities
│   ├── metrics.py
│   └── visualization.py
├── examples/            # Example scripts
│   ├── bakken_workflow_tf.py
│   └── compare_implementations.py
└── tests/               # Unit tests
    ├── test_bnn.py
    └── test_bnn_tf.py
```

## Workflow

The analysis consists of 6 major parts:

1. **Data Reading and Cleaning**: Load well logs, horizons, and 3D seismic data
2. **Cluster Analysis**: Understand relationships between data features
3. **Train/Test Problem**: Create optimal training/testing datasets
4. **Hyperparameter Tuning**: Optimize BNN hyperparameters
5. **3D Prediction**: Predict petrophysical properties across seismic volume
6. **Results Analysis**: Interpret and visualize results

## Usage

### Quick Examples

#### NumPy Implementation
```bash
# Simple example
make example

# Full workflow
make run
```

#### TensorFlow Implementation
```bash
# Simple example
make example-tf

# Full workflow  
make run-tf

# Bakken workflow
make workflow-tf

# Compare implementations
make compare
```

### Using in Python

#### Variational Inference (TensorFlow - Fastest)
```python
from models_tf import BayesianNeuralNetworkTF, VariationalInferenceTF

# Create BNN
bnn = BayesianNeuralNetworkTF(
    topology=[1, 11, 1],
    train_data=train_data,
    test_data=test_data,
    use_gpu=True
)

# Train with VI
vi = VariationalInferenceTF(bnn)
history = vi.train(epochs=500, batch_size=32, verbose=True)

# Predict with uncertainty
predictions = bnn.predict_with_uncertainty(new_data, n_samples=100)
print(f"Mean: {predictions['mean']}")
print(f"Std: {predictions['std']}")
print(f"P10-P90: {predictions['p10']} to {predictions['p90']}")
```

#### MCMC Sampling (TensorFlow - Exact Inference)
```python
from models_tf import BayesianNeuralNetworkTF, MCMCSamplerTF

# Create BNN
bnn = BayesianNeuralNetworkTF([1, 11, 1], train_data, test_data)

# Sample with NUTS (recommended)
sampler = MCMCSamplerTF(bnn, method='nuts')
results = sampler.sample(n_samples=1000, burn_in=500, verbose=True)

# Or use HMC
sampler = MCMCSamplerTF(bnn, method='hmc')
results = sampler.sample(n_samples=1000, burn_in=500)

# Or use Langevin
sampler = MCMCSamplerTF(bnn, method='langevin')
results = sampler.sample(n_samples=1000, burn_in=500)

# Access results
print(f"Acceptance rate: {results['acceptance_rate']:.1f}%")
print(f"Test R²: {np.mean(results['test_metrics']['r2'][-100:]):.4f}")
```

#### NumPy Implementation (Original)
```python
from models import BayesianNeuralNetwork, MCMCSampler

# Create BNN
bnn = BayesianNeuralNetwork([1, 11, 1], train_data, test_data)

# Sample
sampler = MCMCSampler(bnn, use_langevin=True)
results = sampler.sample(n_samples=400, w_step=0.026, tau_step=0.05)
```

## Implementation Comparison

| Feature | NumPy | TensorFlow VI | TensorFlow MCMC |
|---------|-------|---------------|-----------------|
| **Speed** | Slow | Very Fast | Medium |
| **GPU Support** | No | Yes | Yes |
| **Accuracy** | High | Good | High |
| **Uncertainty** | Exact | Approximate | Exact |
| **Scalability** | Poor | Excellent | Good |
| **Best For** | Small data, education | Large data, production | Medium data, exact inference |

### When to Use Which

- **TensorFlow VI**: 
  - ✅ Large datasets (1000+ samples)
  - ✅ Need fast training
  - ✅ GPU available
  - ✅ Approximate uncertainty is acceptable

- **TensorFlow NUTS/HMC**:
  - ✅ Medium datasets (100-1000 samples)
  - ✅ Need exact posterior
  - ✅ GPU available
  - ✅ Can wait longer for results

- **NumPy MCMC**:
  - ✅ Small datasets (<100 samples)
  - ✅ Educational purposes
  - ✅ Understanding the algorithm
  - ✅ No GPU available

## Configuration

All settings are in `config.py`:

### BNN Hyperparameters
```python
BNN_CONFIG = {
    'topology': [1, 11, 1],           # Network architecture
    'learning_rate': 0.01,            # Learning rate
    'w_limit': 0.026874,              # Weight step size
    'tau_limit': 0.0515625,           # Tau step size
    'l_prob': 2.1625,                 # Langevin probability
    'use_langevin': True,
    'n_samples': 400,
    'burn_in_ratio': 0.85,
    'normalization_range': (-1.3875, 1.3875)
}
```

### TensorFlow-Specific Config

See `models_tf/config_tf.py` for:
- GPU settings
- Variational inference parameters
- MCMC method configurations
- Callback settings

## Data Requirements

Place your data files in `rockPropCalculator/Data/`:

1. **WellLogData.xlsx** - Well log data with sheets:
   - Lucy_New
   - Edwards_New
   - Lucy_Seismic_Scaled_New
   - Edwards_Seismic_Scaled_New

2. **Bakken_Horizons_3D.xlsx** - Horizon interpretations

3. **InvertedVol_*.sgy** - 3D seismic volume (P-Impedance)

## Output

Results are saved to `FiguresAndResults/`:

- `training_metrics.png` - Training curves
- `predictions_tf.xlsx` - Predictions with uncertainty
- `bnn_tf_model/` - Saved TensorFlow model
- `weight_posterior.png` - Weight distributions
- TensorBoard logs (for TensorFlow)

## Development

### Running Tests
```bash
# All tests
make test

# Only NumPy tests
make test-numpy

# Only TensorFlow tests
make test-tf

# GPU tests
make test-gpu
```

### Code Quality
```bash
# Format code
make format

# Check linting
make lint

# Type checking
make type-check
```

### Comparing Implementations
```bash
# Run comparison
make compare
```

This will benchmark:
- NumPy MCMC
- TensorFlow VI
- TensorFlow NUTS

And report execution times and accuracy.

## GPU Usage

### Checking GPU Availability
```python
import tensorflow as tf
print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))
```

### Enabling GPU

GPU is enabled by default in TensorFlow implementation. To disable:
```python
bnn = BayesianNeuralNetworkTF(topology, train_data, test_data, use_gpu=False)
```

### Docker GPU Support

To use GPU in Docker:

1. Install NVIDIA Container Toolkit
2. Update `docker-compose.yml`:
```yaml
services:
  bakken-bnn:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

## Advanced Usage

### Custom Training Loop (TensorFlow)
```python
from models_tf import BayesianNeuralNetworkTF

bnn = BayesianNeuralNetworkTF([1, 11, 1], train_data, test_data)

# Custom training
for epoch in range(100):
    # Training step
    losses = bnn.train_step(bnn.x_train, bnn.y_train, kl_weight=0.1)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {losses['total_loss']:.4f}")
```

### Saving and Loading Models
```python
# Save
bnn.save_model('path/to/model')

# Load
bnn.load_model('path/to/model')
```

### Using TensorBoard
```bash
# Start training with TensorBoard logging
# (Logs are automatically created)

# View in browser
tensorboard --logdir=logs/tensorboard

# Open browser to http://localhost:6006
```

## Performance Tips

### For TensorFlow

1. **Use GPU**: 10-100x speedup for large networks
2. **Enable XLA**: Set `xla_compilation=True` in config
3. **Use mixed precision**: Set `mixed_precision=True` for newer GPUs
4. **Batch processing**: Use larger batch sizes with GPU
5. **Reduce samples for VI**: 50-100 forward passes usually sufficient

### For NumPy

1. **Optimize hyperparameters**: Tune acceptance rate to 20-40%
2. **Use Langevin**: Enable gradient-based proposals
3. **Reduce burn-in**: Monitor convergence, adjust as needed
4. **Parallel chains**: Run multiple chains independently

## Troubleshooting

### TensorFlow Issues

**GPU not detected:**
```bash
# Check CUDA installation
nvidia-smi

# Reinstall TensorFlow with GPU support
pip install tensorflow[and-cuda]>=2.12.0
```

**Out of memory:**
```python
# Enable memory growth in config_tf.py
TF_CONFIG = {
    'memory_growth': True
}

# Or reduce batch size
vi.train(epochs=500, batch_size=16)
```

**Slow training:**
```python
# Enable XLA compilation
TF_CONFIG = {
    'xla_compilation': True
}

# Use mixed precision (requires GPU with Tensor Cores)
TF_CONFIG = {
    'mixed_precision': True
}
```

### NumPy Issues

**Low acceptance rate (<20%):**
- Decrease `w_limit` and `tau_limit` in config

**High acceptance rate (>60%):**
- Increase `w_limit` and `tau_limit` in config

**Poor convergence:**
- Increase `n_samples`
- Adjust `learning_rate`
- Try different `topology`

## Examples

### 1. Quick Start
```bash
# TensorFlow VI (fastest)
make example-tf

# NumPy MCMC (educational)
make example
```

### 2. Compare Methods
```bash
make compare
```

This runs:
- NumPy MCMC
- TensorFlow VI
- TensorFlow NUTS

And shows speed/accuracy comparison.

### 3. Full Bakken Workflow
```bash
make workflow-tf
```

## API Reference

### BayesianNeuralNetworkTF
```python
class BayesianNeuralNetworkTF:
    def __init__(self, topology, train_data, test_data, 
                 learning_rate=0.01, use_gpu=True)
    
    def predict_with_uncertainty(self, x, n_samples=100)
    # Returns: {'mean', 'std', 'p10', 'p50', 'p90', 'samples'}
    
    def save_model(self, filepath)
    def load_model(self, filepath)
```

### MCMCSamplerTF
```python
class MCMCSamplerTF:
    def __init__(self, bnn, method='nuts', use_langevin=True)
    # method: 'hmc', 'nuts', or 'langevin'
    
    def sample(self, n_samples=1000, burn_in=500, 
               step_size=0.01, verbose=False)
    # Returns: {'weight_samples', 'train_metrics', 
    #           'test_metrics', 'acceptance_rate'}
```

### VariationalInferenceTF
```python
class VariationalInferenceTF:
    def __init__(self, bnn)
    
    def train(self, epochs=1000, batch_size=32, 
              kl_weight=1.0, verbose=True)
    # Returns: history dict with loss and metrics
    
    def predict_with_uncertainty(self, x, n_samples=100)
```

## Performance Benchmarks

Tested on synthetic data (n=100):

| Method | Time | Test R² | Uncertainty |
|--------|------|---------|-------------|
| NumPy MCMC | 45s | 0.89 | Exact |
| TF VI | 3s | 0.86 | Approximate |
| TF NUTS | 12s | 0.89 | Exact |
| TF HMC | 15s | 0.88 | Exact |

*Note: Times on CPU. GPU can be 10-50x faster for TensorFlow.*

## Well Locations

### Lucy Well
- **Location**: Inline 2108, Xline 1165
- **Bakken Tops (MD_KB_ft)**:
  - BKKNU: 8718.0
  - BKKNM: 8733.0
  - BKKNL: 8775.5
  - TRFK: 8808.5

### Edwards Well
- **Location**: Inline 1824, Xline 1073
- **Bakken Tops (MD_KB_ft)**:
  - BKKNU: 9040.0
  - BKKNM: 9057.5
  - BKKNL: 9104.0
  - TRFK: 9136.0

## References

### Methodological References

1. **Bayesian Neural Networks**: Neal, R. M. (1996). Bayesian Learning for Neural Networks
2. **Langevin Dynamics**: Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence of Langevin distributions
3. **Variational Inference**: Blei, D. M., et al. (2017). Variational Inference: A Review for Statisticians
4. **Hamiltonian Monte Carlo**: Neal, R. M. (2011). MCMC using Hamiltonian dynamics
5. **NUTS**: Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler

### Software References

- TensorFlow: https://www.tensorflow.org/
- TensorFlow Probability: https://www.tensorflow.org/probability
- scikit-learn: https://scikit-learn.org/
- segyio: https://github.com/equinor/segyio

## Citation

If you use this code in your research, please cite:
```
Tomski, J. R. (2025). Unconventional Reservoir Characterization utilizing 
Bayesian Neural Learning via Langevin Dynamics: A Bakken Case Study. 
Master's Thesis, University of Texas at Austin.
```

## License

Academic use - University of Texas at Austin

## Contributing

This is research code. For issues or questions, please contact the author.

## Changelog

### Version 1.0.0
- Initial release with NumPy implementation
- TensorFlow implementation added
- Variational Inference support
- Multiple MCMC methods (HMC, NUTS, MALA)
- GPU acceleration
- Comprehensive testing suite
- Docker containerization

## Acknowledgments

- University of Texas at Austin - Jackson School of Geosciences
- Institute for Geophysics
- Advisors and committee members