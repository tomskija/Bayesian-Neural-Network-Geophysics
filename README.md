# Bayesian Neural Network for Rock Property Predictions

Master's thesis code from UT Austin — applying Bayesian Neural Networks to unconventional reservoir characterization on Bakken field data. The goal is to predict petrophysical properties from well log and 3D seismic data while producing calibrated uncertainty estimates alongside each prediction.

## What is this?

Two BNN implementations are included depending on what you need:

| Implementation | Speed | Uncertainty | Best For |
|---|---|---|---|
| **NumPy** (`models/`) | Slow | Exact | Learning the algorithm, small datasets |
| **TensorFlow** (`models_tf/`) | Fast, GPU-ready | Exact or approximate | Production, larger datasets |

The TensorFlow implementation supports three inference methods — Variational Inference (fastest), NUTS, and HMC. The NumPy version uses Langevin MCMC and is the original thesis implementation.

## Project Structure

```
Bayesian-Neural-Network-for-Rock-Property-Predictions/
├── .devcontainer/              # VS Code dev container config
├── rockPropCalculator/
│   ├── config.py               # hyperparameters and constants
│   ├── main.py                 # NumPy workflow entry point
│   ├── main_tf.py              # TensorFlow workflow entry point
│   ├── Data/                   # input data files (well logs, seismic)
│   ├── FiguresAndResults/      # output plots and predictions
│   ├── data/                   # data loading and processing
│   ├── models/                 # NumPy BNN + MCMC
│   ├── models_tf/              # TensorFlow BNN + VI + MCMC
│   └── utils/                  # metrics and visualization
├── tests/
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

## Getting Started

### VS Code Dev Container (easiest)

```bash
git clone git@github.com:tomskija/Bayesian-Neural-Network-for-Rock-Property-Predictions.git
cd Bayesian-Neural-Network-for-Rock-Property-Predictions
code .
```

Hit `F1` → `Dev Containers: Reopen in Container`, then:

```bash
cd rockPropCalculator
python main_tf.py   # TensorFlow workflow
python main.py      # NumPy workflow
```

### Docker Compose

```bash
docker-compose up -d
docker exec -it bakken-bnn-dev bash
cd rockPropCalculator && python main_tf.py
```

### Local Python

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
cd rockPropCalculator && python main_tf.py
```

**Optional GPU support:**
```bash
pip install tensorflow[and-cuda]>=2.12.0
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Data

Place your data files in `rockPropCalculator/Data/`:

- `WellLogData.xlsx` — well log data (Lucy and Edwards wells)
- `Bakken_Horizons_3D.xlsx` — horizon interpretations
- `InvertedVol_*.sgy` — 3D seismic volume (P-Impedance)

Results and figures are written to `FiguresAndResults/`.

## Running the Workflow

The Makefile handles most common tasks:

```bash
# TensorFlow (recommended)
make example-tf     # quick example
make run-tf         # full workflow
make workflow-tf    # full Bakken workflow
make compare        # benchmark NumPy vs TF

# NumPy (original thesis implementation)
make example        # quick example
make run            # full workflow
```

## Quick Code Examples

**TensorFlow — Variational Inference (fastest)**
```python
from models_tf import BayesianNeuralNetworkTF, VariationalInferenceTF

bnn = BayesianNeuralNetworkTF(topology=[1, 11, 1], train_data=train_data, test_data=test_data)
vi = VariationalInferenceTF(bnn)
vi.train(epochs=500, batch_size=32)

predictions = bnn.predict_with_uncertainty(new_data, n_samples=100)
# returns mean, std, p10, p50, p90
```

**TensorFlow — MCMC (exact inference)**
```python
from models_tf import BayesianNeuralNetworkTF, MCMCSamplerTF

bnn = BayesianNeuralNetworkTF([1, 11, 1], train_data, test_data)
sampler = MCMCSamplerTF(bnn, method='nuts')  # or 'hmc', 'langevin'
results = sampler.sample(n_samples=1000, burn_in=500)
```

**NumPy — original MCMC**
```python
from models import BayesianNeuralNetwork, MCMCSampler

bnn = BayesianNeuralNetwork([1, 11, 1], train_data, test_data)
sampler = MCMCSampler(bnn, use_langevin=True)
results = sampler.sample(n_samples=400, w_step=0.026, tau_step=0.05)
```

## Testing & Code Quality

```bash
make test           # all tests
make test-numpy     # NumPy only
make test-tf        # TensorFlow only
make format         # black formatting
make lint           # flake8
```

## Troubleshooting

**GPU not detected** — run `nvidia-smi` to confirm CUDA is installed, then `pip install tensorflow[and-cuda]>=2.12.0`

**Out of memory** — set `memory_growth: True` in `models_tf/config_tf.py` or reduce batch size

**Low MCMC acceptance rate (<20%)** — decrease `w_limit` and `tau_limit` in `config.py`

**High acceptance rate (>60%)** — increase `w_limit` and `tau_limit`

## Citation

```
Tomski, J. R. (2025). Unconventional Reservoir Characterization utilizing 
Bayesian Neural Learning via Langevin Dynamics: A Bakken Case Study. 
Master's Thesis, University of Texas at Austin.
```

## Questions / Issues

Open an issue [here](https://github.com/tomskija/Bayesian-Neural-Network-for-Rock-Property-Predictions/issues).
