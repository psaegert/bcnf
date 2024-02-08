<h1 align="center">
    <img style="width: 150px" src="bcnf_icon.png" alt="Icon">
</h1>


<h1 align="center" style="margin-top: 0px;">BCNF: Ballistic Conditional Normalizing Flows</h1>
<h2 align="center" style="margin-top: 0px;">Generative Neural Networks for the Sciences: Final Project</h2>

<!-- <div align="center">

[![pytest](https://github.com/psaegert/elqm-INLPT-WS2023/actions/workflows/pytest.yml/badge.svg)](https://github.com/psaegert/elqm-INLPT-WS2023/actions/workflows/pytest.yml)
[![quality checks](https://github.com/psaegert/elqm-INLPT-WS2023/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/psaegert/elqm-INLPT-WS2023/actions/workflows/pre-commit.yml)

</div> -->


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
bcnf demo --dummy_option "Hello World"
```

**Python API**
```python
from bcnf import ...
```


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
If you use ELQM: Energy-Law Query-Master for your research, please cite it using the following

```bibtex
@software{bcnf2024,
    author = {Christian Kleiber and Paul Saegert and Nikita Tatsch},
    title = {BCNF: Ballistic Conditional Normalizing Flows},
    month = mar,
    year = 2024,
    publisher = {GitHub},
    version = {0.0.1},
    url = {https://github.com/MrWhatZitToYaa/IGNNS-final-project}
}
```