# Airplane Boarding Simulation

An agent-based, discrete time-step simulation comparing four airplane boarding
strategies on an Airbus A320, under ideal literature conditions and under
Philippine domestic flight conditions.

Course project for Modeling and Simulation.

## What this repository contains

| Folder | What is in it |
|---|---|
| `boarding_sim/` | The simulation itself: the Python package, experiment scripts, tests, web dashboard and generated results |
| `presentation/` | The slide deck |
| `references/` | The published papers the model is validated against |

## Quick start

```bash
cd boarding_sim
pip install -r requirements.txt
pytest                      # 16 tests, about 3 seconds
python3 run_baseline.py     # Experiment 1, writes results/exp1_baseline.csv
python3 run_experiments.py  # Experiments 2 to 6, about 15 minutes
python3 web_app.py          # interactive dashboard at http://localhost:5050
```

`boarding_sim/README.md` documents every script, parameter and output in detail,
including how to reproduce each figure.

## Results in one line

Steffen is the fastest strategy in every condition tested, but Philippine
passenger behaviour costs it 76 percent of its performance, so boarding-call
discipline matters more than the choice of strategy.

Baseline means across 200 replications: Steffen 687 s, Outside-In 1,050 s,
Random 1,238 s, Back-to-Front 1,594 s.
