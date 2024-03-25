<h1 align="center">
    <img style="width: 150px" src="./images/bcnf-icon.webp" alt="Icon">
</h1>


<h1 align="center" style="margin-top: 0px;">BCNF: Ballistic Conditional Normalizing Flows</h1>
<h2 align="center" style="margin-top: 0px;">Generative Neural Networks for the Sciences: Final Project</h2>

<div align="center">

[![pytest](https://github.com/MrWhatZitToYaa/IGNNS-final-project/actions/workflows/pytest.yml/badge.svg)](https://github.com/MrWhatZitToYaa/IGNNS-final-project/actions/workflows/pytest.yml)
[![quality checks](https://github.com/MrWhatZitToYaa/IGNNS-final-project/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/MrWhatZitToYaa/IGNNS-final-project/actions/workflows/pre-commit.yml)

</div>


# Introduction

# Requirements

## Hardware
- 8GB RAM
- GPU (optional)

## Software
- Python 3.11
- `pip` >= [21.3](https://pip.pypa.io/en/stable/news/#v21-3)

# Getting Started
## 1. Clone the repository

```sh
git clone https://github.com/MrWhatZitToYaa/IGNNS-final-project
cd IGNNS-final-project
```

## 2. Install the package

Optional: Create a virtual environment:

**conda:**

```sh
conda create -n bcnf python=3.11 [ipykernel]
conda activate bcnf
```

**venv:**

```bash
python3 -m venv bcnf_venv
source bcnf_venv/bin/activate
```

Then, install the package via

```sh
pip install -e .
```

# Usage

**CLI**

```sh
bcnf train -c configs/runs/trajectory_LSTM_large.yaml
```


**Python API**

Load a model from a configuration file:
```python
import os
import json
import torch
from bcnf import CondRealNVP_v2

MODEL_NAME = "trajectory_LSTM_large"

# Load the config
with open(os.path.join(get_dir('models', 'bcnf-models', MODEL_NAME), 'config.json'), 'r') as f:
    config = load_config(json.load(f)['config_path'])

# Load the model
model = CondRealNVP_v2.from_config(config).to(device)
cnf.load_state_dict(torch.load(os.path.join(get_dir('models', 'bcnf-models', MODEL_NAME), "state_dict.pt")))
cnf.eval()
```

Train your own model:
```python
from bcnf.utils import get_dir, load_config, sub_root_path
from bcnf.train import Trainer
from bcnf import CondRealNVP_v2

# Specify a path template for the config file
config_path_pattern = os.path.join("{{BCNF_ROOT}}", "configs", "runs", "my_config.yaml")

# Find the config file in the local filesystem
config_path = sub_root_path(config_path_pattern)

# Load the config and create a model
config = load_config(config_path, verify=False)
model = CondRealNVP_v2.from_config(config).to(device)

# Create a Trainer instance and load the data specified in the config
trainer = Trainer(
    config={k.lower(): v for k, v in config.to_dict().items()},
    project_name="bcnf-test",  # Name of the Weights & Biases project
    parameter_index_mapping=model.parameter_index_mapping,
    verbose=True,
)

# Train
model = trainer.train(model)

# Save
torch.save(model.state_dict(), os.path.join(get_dir('models', 'bcnf-models', MODEL_NAME, create=True), f"state_dict.pt"))

with open(os.path.join(get_dir('models', 'bcnf-models', MODEL_NAME, create=True), 'config.json'), 'w') as f:
    json.dump({'config_path': config_path_pattern}, f)
```


# Train the model
model = trainer.train(model)


# Development

## Setup
To set up the development environment, run the following commands:

```sh
pip install -e .[dev]
pre-commit install
```

## Tests

To run the tests locally, run the following commands:

```sh
pytest tests --cov src
```

# Citation
If you use our work for your research, please cite it using the following

```bibtex
@software{bcnf2024,
    author = {Christian Kleiber and Paul Saegert and Nikita Tatsch},
    title = {BCNF: Ballistic Conditional Normalizing Flows},
    month = mar,
    year = 2024,
    publisher = {GitHub},
    version = {0.1.0},
    url = {https://github.com/MrWhatZitToYaa/IGNNS-final-project}
}
```