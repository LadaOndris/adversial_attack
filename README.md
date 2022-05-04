
## Adversarial Attacks via Genetic Algorithm

Author: Ladislav Ondris

## How to run
Prepare environment:
```
conda env create -f environment.yml
conda activate evo
pip install pygad
```
Export path to the project root directory:
```
export PYTHONPATH="${PYTHONPATH}:<PROJECT_DIR>"
```
### Run experiments

```
python3 src/experiments/runner.py
```

### Display plots from experiments

```
python3 src/experiments/plots.py
```



