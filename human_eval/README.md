# HumanEval: Hand-Written Evaluation Set 

## Installation

Install human-eval repository
```
$ pip install -e human-eval
```


Also needs to be installed
```
$ git clobe https://github.com/Deelvin/lm-evaluation-harness.git
$ cd lm-evaluation-harness
$ pip install -e .
```


## Samples generation

To generate samples for each endpoints run:
```
$ python generate_samples.py
```

This script generates a results folder, where for each endpoint there will be generated samnples and metrics


To display incorrect samples, the verification script *samples.py* can be used. This script generates Python scripts for incorrect samples and is so convenient for debugging model outputs.


To run these wrong scripts, the *run wrong.py* script can be used to run one after the other.

### Manual evaluation

To calculate metrics manually:
```
$ evaluate_functional_correctness $PATH_TO_SAMPLES
```