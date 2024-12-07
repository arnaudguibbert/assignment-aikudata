# Assaiku library

The `assaiku`library allows you to perform several tasks on the data coming from the ``Current Population Survey``. Here are the tasks you can perform with this library:
- Data Exploration.
- Data Cleaning and preprocessing.
- Building ML models and evaluating them.

## Dependencies

- This program is designed to work on Unix systems (Linux distributions/Mac OS)
- conda or micromamba
- make
- libomp (used by xgboost)

## Getting started

### Environment setup

Python package dependencies will be installed in an isolated conda environment, in addition a jupyter kernel will be created for jupyter notebook support. To do so, run the following make command from the repository root. Replace `<MANAGER>` by your package manager (`conda` or `micromamba`).

```
make py-setup-env-dev MANAGER=<MANAGER>
```

It creates an environment named `assaiku`, a kernel name `assaiku (Python 3.10)` and installs librairies. 

### Data import

Please download the data from [here](https://drive.google.com/drive/folders/1PPsjCoM130k3n3V4roq-yF74jkPjkVd7) and put it into a `data/raw` folder as follows:
```
data
└── raw
    ├── census_income_learn.csv
    ├── census_income_metadata.txt
    └── census_income_test.csv
```
The ``data`` folder should be located at the root of the repository.

## Reproduce results

To reproduce the results, you have two options:

### Notebook option
The first option consists in running sequentially the different notebooks located [here](./notebooks/). It gives details and explanations about the operations we are performing on the data. There are three notebooks (one for each phase):
- [Data Exploration](/notebooks/data_exploration.ipynb)
- [Data Cleaning & Processing](/notebooks/data_exploration.ipynb)
- [ML Exploration](/notebooks/ml_pipeline.ipynb)

### CLI option

The second option is to run a command through the CLI. It will generate all results presented in the [results](./results/) folder. To do so, run this command in the `assaiku` environment at the root of the repository:
```
assaiku
```
It runs the Data Pipeline, as well as the ML Pipeline.

## Our findings

All our findings are reported in this [presentation](./presentation.pdf)