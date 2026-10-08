# Airplane Boarding Strategy Simulation
**Course:** Modeling and Simulation (University Project)
**Model:** Airbus A320, Philippine Domestic Flight, Single Front Door

---

## Overview

A time-step, agent-based simulation comparing four boarding strategies and
studying how Filipino passenger behaviour (group travel, heavy hand-carry,
low compliance, etc.) changes each strategy's performance.

**Research questions**
1. Which strategy minimises total boarding time under ideal (literature) conditions?
2. How do group travel and low boarding-call compliance change strategy rankings?
3. Which strategy is most robust for a Philippine domestic flight?

---

## Project Structure

```
boarding_sim/
├── config.yaml              # ALL parameters – edit here, never in code
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── boarding/
│   ├── __init__.py          # Package re-exports
│   ├── cabin.py             # Cabin layout, seat map, blocking logic
│   ├── passenger.py         # Passenger dataclass + states
│   ├── strategies.py        # 4 boarding-order functions
│   ├── behavior.py          # Filipino behaviour modifiers (7 features)
│   ├── simulation.py        # Core time-step engine (dt = 1 s)
│   ├── metrics.py           # Per-run metric collection
│   ├── experiments.py       # Replications, CRN, sweeps
│   ├── stats.py             # Paired t-test, Mann-Whitney, ANOVA, rankings
│   └── animate.py           # Matplotlib GIF animation
├── tests/
│   ├── test_single_passenger.py   # Hand-computed verification tests
│   ├── test_two_same_row.py       # Seat interference tests
│   ├── test_strategies.py         # Strategy permutation tests
│   └── test_invariants.py         # Engine invariant tests
├── run_baseline.py          # Experiment 1 (baseline comparison)
├── run_experiments.py       # Experiments 2-6 (sweeps + ablation)
├── results/                 # CSVs and PNG charts written here
└── analysis.ipynb           # Jupyter analysis notebook
```

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10 or newer is required.

---

## How to Run

### 1. Tests (run first to verify your environment)
```bash
pytest
```
Expected: **16 passed**.

### 2. Baseline comparison (Experiment 1)
```bash
python run_baseline.py
```
Produces `results/exp1_baseline.csv` and `results/fig1_boxplot_baseline.png`.

### 3. Philippine scenario (Experiment 2)
```bash
python run_experiments.py --exp 2
```

### 4. All sweep experiments (Experiments 2–6)
```bash
python run_experiments.py
```
Runs all six experiments. Takes ~15 minutes on a normal laptop (200 reps × 4 strategies per condition).

### 5. Single experiment
```bash
python run_experiments.py --exp 3,5    # only compliance + sensitivity sweeps
```

### 6. Boarding animation
```bash
# One strategy (default: steffen, baseline scenario)
python boarding/animate.py --strategy steffen --scenario baseline --seed 12345

# All four strategies, Philippine scenario
python boarding/animate.py --strategy all --scenario philippine

# See all options
python boarding/animate.py --help
```
Saves `results/anim_<strategy>_<scenario>.gif`.

### 7. Analysis notebook
```bash
jupyter notebook analysis.ipynb
```
Run all cells from the `boarding_sim/` directory.

---

## Cabin Model (Airbus A320)

- **30 rows**, seats A B C | aisle | D E F → **180 seats**, fully booked.
- **Row pitch:** 0.79 m (ASSUMED; literature range 0.75–0.81 m).
- Aisle modelled as a 1-D array of 30 cells. Cell 0 = front (door), cell 29 = rear.
- **Seat blocking (left side):** C blocks B, B blocks A.
  Mirror on the right: D blocks E, E blocks F.

---

## Boarding Strategies

| # | Strategy | Description |
|---|---|---|
| 1 | **Random** | Passengers board in uniformly random order. |
| 2 | **Back-to-Front** | 5 zones of 6 rows; zone 1 (rows 25–30) boards first. Random within zone. |
| 3 | **Outside-In (WilMA)** | Window → Middle → Aisle; random within each wave. |
| 4 | **Steffen** | Even rows then odd rows, back-to-front per wave; alternates left/right. Follows Steffen (2008). |

---

## Filipino Behaviour Features (`config.yaml`)

All features are **off** in the baseline scenario and **on** in the
Philippine scenario. Each has an `enabled` flag.

| Feature | What it does |
|---|---|
| `group_travel` | Groups of 2–4 board together at the earliest-called member's position. `group_fraction = 0.40` |
| `non_compliance` | 30% of passengers ignore their call and jump to the first 30% of the queue. `compliance_rate = 0.70` |
| `late_passengers` | 3% of passengers are moved to the end of the queue. |
| `heavy_handcarry` | More passengers carry 2+ bags (heavier pasalubong culture). 2-bag probability increases from 25% → 50%. |
| `bayanihan` | A group mate or the next passenger behind helps stow bags; cuts stow time by 35%. `p_help = 0.15` |
| `seat_search_delay` | 10% of passengers take extra time (Uniform 5–15 s) to find their row. |
| `seat_swapping` | Optional (off by default): some group members swap seats. |

> **All behaviour parameter values are ASSUMED** (no published Filipino aviation data).
> Label each assumption clearly in any report.

---

## Key Results (200 replications × 4 strategies)

### Baseline scenario (ideal conditions)
| Strategy | Mean (s) | Rank |
|---|---|---|
| **Steffen** | **687** | 1 |
| Outside-In | 1050 | 2 |
| Random | 1238 | 3 |
| Back-to-Front | 1594 | 4 |

Ranking consistent with Steffen (2008) and Milne & Kelly. All pairwise
differences are statistically significant (paired t-test, Bonferroni-corrected,
all p < 0.001; F(3, 796) = 8853, p < 0.001 by one-way ANOVA).

### Philippine scenario
| Strategy | Mean (s) | vs Baseline |
|---|---|---|
| Steffen | 1211 | +76% |
| Outside-In | 1384 | +32% |
| Random | 1455 | +18% |
| Back-to-Front | 1736 | +9% |

Steffen remains fastest but suffers the largest absolute slowdown because
structured spacing breaks down when passengers clump in groups and ignore
call order.

### Strategy ranking – robust under sensitivity analysis
Steffen stays ranked #1 when stow time is doubled (×2.0) or walk speed
is reduced to 0.6 m/s. Back-to-Front stays last in all 12 conditions.

---

## Model Assumptions

All ASSUMED parameters are in `config.yaml` with `# ASSUMED` comments.
Key assumptions:

| Parameter | Value | Source |
|---|---|---|
| Time step | 1 s | ASSUMED |
| Entry interval | 3 s between passengers | ASSUMED |
| Walk speed | Normal(0.8, 0.15) m/s, min 0.4 | ASSUMED |
| Stow time | Lognormal(mean=9 s, sd=4 s) per bag | ASSUMED |
| Base seat time | 2 s (no blockers) | ASSUMED |
| Shuffle time | Uniform(4, 10) s per blocking passenger | ASSUMED |
| Row pitch | 0.79 m | ASSUMED (lit. 0.75–0.81) |
| Group fraction | 0.40 | ASSUMED |
| Compliance rate | 0.70 | ASSUMED |
| Late fraction | 0.03 | ASSUMED |

**Do not cite these as published literature.** They are reasonable engineering
defaults chosen to produce qualitatively correct behaviour. Fill in real
citations before submitting a formal report.

---

## Validation

Expected qualitative results (baseline, based on published boarding research):
- Random ≤ Back-to-Front (B2F is no faster than random, often worse). ✅
- Outside-In and Steffen are the fastest strategies. ✅

`run_baseline.py` prints a warning if results contradict these expectations.

---

## Reproducibility

Every run is seeded. The same `--seed` always gives the same output.
For the experiment suite, replication *i* uses `base_seed + i` (default 12345).

To reproduce all results from scratch:
```bash
pip install -r requirements.txt
pytest                         # verify environment
python run_baseline.py         # Experiment 1
python run_experiments.py      # Experiments 2-6
jupyter notebook analysis.ipynb
```

---

## Acknowledgements

Developed as a Modeling and Simulation university course project.
Boarding strategy literature: Steffen (2008). Cabin model based on Airbus A320 documentation.
