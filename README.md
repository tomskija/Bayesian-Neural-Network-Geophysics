# Bayesian-Neural-Network-Geophysics

A Python project applying Bayesian Neural Networks to unconventional reservoir characterization on Bakken field data. Rooted in my UT Austin Master's thesis and peer-reviewed publication in AAPG Bulletin — actively being extended and improved beyond the original work.

The core idea: predict petrophysical properties (total porosity, TOC) from well log and 3D seismic data using MCMC via Langevin dynamics, while producing calibrated uncertainty estimates (mean, P10, P90) alongside each prediction.

## What is this?

The BNN uses Markov Chain Monte Carlo via Langevin dynamics to sample from the posterior distribution — this gives you uncertainty estimates, not just point predictions. The workflow integrates prestack seismic inversion output with well log data to predict reservoir properties spatially across the 3D seismic volume.

## Project Structure

```
Bayesian-Neural-Network-Geophysics/
├── .devcontainer/              # VS Code dev container config
├── rockPropCalculator/
│   ├── config.py               # hyperparameters and constants
│   ├── main.py                 # main workflow entry point
│   ├── Data/                   # input data files (well logs, seismic)
│   ├── FiguresAndResults/      # output plots and predictions
│   ├── data/                   # data loading and processing
│   ├── models/                 # BNN + MCMC implementation
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
git clone git@github.com:tomskija/Bayesian-Neural-Network-Geophysics.git
cd Bayesian-Neural-Network-Geophysics
code .
```

Hit `F1` → `Dev Containers: Reopen in Container`, then:

```bash
cd rockPropCalculator
python main.py
```

### Docker Compose

```bash
docker-compose up -d
docker exec -it bakken-bnn-dev bash
cd rockPropCalculator && python main.py
```

### Local Python

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
cd rockPropCalculator && python main.py
```

## Data

Place your data files in `rockPropCalculator/Data/`:

- `WellLogData.xlsx` — well log data (Lucy and Edwards wells)
- `Bakken_Horizons_3D.xlsx` — horizon interpretations
- `InvertedVol_*.sgy` — 3D seismic volume (P-Impedance)

Results and figures are written to `FiguresAndResults/`.

## Running the Workflow

```bash
make example    # quick example
make run        # full workflow
make test       # run tests
make format     # black formatting
make lint       # flake8
```

## Quick Code Example

```python
from models import BayesianNeuralNetwork, MCMCSampler

bnn = BayesianNeuralNetwork([1, 11, 1], train_data, test_data)
sampler = MCMCSampler(bnn, use_langevin=True)
results = sampler.sample(n_samples=400, w_step=0.026, tau_step=0.05)
# predictions include mean, P10, P90 uncertainty estimates
```

## Troubleshooting

**Low MCMC acceptance rate (<20%)** — decrease `w_limit` and `tau_limit` in `config.py`

**High acceptance rate (>60%)** — increase `w_limit` and `tau_limit`

**Poor convergence** — increase `n_samples` or try adjusting the network topology in `config.py`

## Citation

This repo builds on the following work:

**Journal Publication**
```
Tomski, J.R., Sen, M.K., Hess, T.E., Pyrcz, M.J. (2022). Unconventional reservoir 
characterization by seismic inversion and machine learning of the Bakken Formation. 
AAPG Bulletin, 106(11): 2203–2223. https://doi.org/10.1306/12162121035
```

**Master's Thesis**
```
Tomski, J.R. (2020). Unconventional reservoir parameter estimation by seismic inversion 
and machine learning of the Bakken Formation, North Dakota. 
Master's Thesis, University of Texas at Austin.
```

## Questions / Issues

Open an issue [here](https://github.com/tomskija/Bayesian-Neural-Network-Geophysics/issues).
