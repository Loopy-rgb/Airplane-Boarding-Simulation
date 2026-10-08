# PRD: Airplane Boarding Strategy Simulation (Philippine Domestic Flight)

> **Instructions to the AI building this:** Build the project exactly as specified below. Ask me before deviating. Where a parameter is marked `[ASSUMED]`, use the given default, keep it in a single config file, and never hard-code it elsewhere. Write clean, commented code that a university student can read and explain in a presentation. Build in the milestone order in Section 12 and stop after each milestone so I can run it.

---

## 1. Project Overview

**Course:** Modeling and Simulation (university project, 5 weeks)
**Goal:** Build a time-step, agent-based simulation of passengers boarding an Airbus A320 through a single front door, compare four boarding strategies, and test how each strategy holds up under Filipino passenger behavior (group travel, ignoring boarding calls, heavy hand-carry, etc.).

**Research questions**
1. Which strategy minimizes total boarding time under ideal (literature) conditions?
2. How do group travel and low boarding-call compliance change each strategy's performance and ranking?
3. Which strategy is the most robust for a Philippine domestic flight?

**Primary outputs:** a runnable Python simulation, experiment results (CSV + charts), an animation of a boarding run, and a results summary usable in a written report.

---

## 2. Tech Stack and Constraints

- **Language:** Python 3.10+
- **Libraries:** `numpy`, `pandas`, `scipy`, `matplotlib`. `pyyaml` for config. Optional: `tqdm`. **No SimPy** and no heavy frameworks.
- **Interface:** command-line scripts plus one Jupyter notebook for analysis. No web app needed.
- **Determinism:** every run must be reproducible from a seed.
- **Performance:** one replication should finish in under 1 second; the full experiment suite (Section 8) should finish in a few minutes on a normal laptop.

---

## 3. Project Structure

```
boarding_sim/
├── config.yaml              # all parameters (baseline + Filipino scenario)
├── requirements.txt
├── README.md                # how to install and run
├── boarding/
│   ├── __init__.py
│   ├── cabin.py             # cabin layout, seat map
│   ├── passenger.py         # Passenger dataclass + states
│   ├── strategies.py        # 4 ordering functions
│   ├── behavior.py          # Filipino behavior modifiers
│   ├── simulation.py        # core time-step engine
│   ├── metrics.py           # per-run metric collection
│   ├── experiments.py       # replications, common random numbers, sweeps
│   ├── stats.py             # CI, paired t-test, ANOVA
│   └── animate.py           # matplotlib animation of one run
├── tests/
│   ├── test_single_passenger.py
│   ├── test_two_same_row.py
│   ├── test_strategies.py
│   └── test_invariants.py
├── run_baseline.py          # baseline comparison
├── run_experiments.py       # all experiments in Section 8
├── results/                 # CSVs and PNG charts written here
└── analysis.ipynb
```

---

## 4. Cabin Model

- **Aircraft:** Airbus A320, **30 rows**, seats **A B C | aisle | D E F**, so **180 seats**, fully booked (180 passengers).
- **Row pitch:** `0.79 m` `[ASSUMED, literature range 0.75-0.81]`.
- **Aisle:** 1D array of cells. Use **1 cell per row**, indexed `0` (door / front) to `29` (rear row). Row 1 is at cell 0. A queue area sits before the door (not part of the cell array).
- **Seat map:** dict or array `occupied[row][letter]`. Seat letters A, B, C are the left block (A = window, C = aisle side). D, E, F are the right block (F = window, D = aisle side).
- **Seat types:** window = A, F; middle = B, E; aisle = C, D.
- **Blocking classes:** on the left, C blocks B blocks A (a passenger to A must pass C and B). Mirror on the right: D blocks E blocks F.

---

## 5. Passenger Agent

Dataclass fields:

| Field | Description |
|---|---|
| `id` | unique int |
| `row`, `letter` | assigned seat |
| `group_id` | int or `None` |
| `num_bags` | 0, 1, 2 or more hand-carry items |
| `walk_speed` | m/s, drawn per passenger |
| `stow_time` | seconds, drawn from bag count |
| `seat_search_delay` | extra seconds at the row (0 for most) |
| `is_late` | bool |
| `is_compliant` | bool, whether they board in their assigned call order |
| `entry_time` | tick at which they enter the aisle |
| `state` | see below |
| `aisle_cell` | current cell or `None` |
| `wait_ticks`, `aisle_interference_events`, `seat_interference_events` | counters |
| `seated_time` | tick when seated |

**States:** `QUEUED -> WALKING -> BLOCKED -> STOWING -> SEATING -> SEATED`

---

## 6. Simulation Engine Rules

Time-step simulation with **`dt = 1 second`** ticks `[ASSUMED]`. Update passengers **from front-most in the aisle to rear-most** each tick (or use a two-phase move: decide moves, then apply) so that no two passengers ever occupy the same cell.

**Entry:** every `entry_interval` seconds (default `3 s`, `[ASSUMED]`) the next passenger in the queue enters cell 0 **only if cell 0 is free**. If it is not free, they wait in the queue.

**Walking:** a passenger in the aisle advances one cell when:
- the next cell is empty, **and**
- a movement check passes. Cell traversal time is `row_pitch / walk_speed`; implement it as an accumulated-distance counter (progress += `walk_speed * dt`; move when progress >= `row_pitch`), or an equivalent approach. Faster passengers must never pass slower ones (the aisle is single file).

If the next cell is occupied, the passenger becomes `BLOCKED`, `wait_ticks += 1`, and **one aisle interference event** is counted per continuous blocked episode (not per tick).

**Stowing:** upon reaching their assigned row cell, the passenger enters `STOWING` for `stow_time + seat_search_delay` ticks. They **occupy the aisle cell and block everyone behind them** for that time.

**Seating:**
- Check whether any seat between the passenger's seat and the aisle is already occupied (e.g., passenger to A with B or C occupied).
- If clear: `SEATING` takes `base_seat_time` (default `2 s`), then `SEATED` and the aisle cell is freed.
- If blocked by seated passengers: add `shuffle_time` per blocking passenger (drawn from `Uniform(4, 10)` s per blocker `[ASSUMED]`; if both B and C block A, apply it twice). Count **one seat interference event per blocking passenger**. The passenger stays in the aisle cell during the shuffle.

**Termination:** when all 180 passengers are `SEATED`. `total_boarding_time = max(seated_time)`. Add a safety cap (e.g., 5000 ticks) that raises an error if hit (deadlock detection).

**Invariants (must hold every tick, assert in test mode):**
1. No two passengers share an aisle cell.
2. A passenger never moves backward.
3. Each seat holds at most one passenger.
4. A passenger is seated only in the assigned seat (or in a swapped seat if the swap feature is on).

---

## 7. Boarding Strategies (`strategies.py`)

Each strategy is a function `order(passengers, rng) -> list[Passenger]` returning the call order. Row 1 is front, row 30 is back.

1. **Random:** uniform shuffle.
2. **Back-to-front (zones):** 5 zones of 6 rows each. Zone 1 = rows 25-30, zone 2 = rows 19-24, ..., zone 5 = rows 1-6. Zones called in that order; **random order within a zone**. Zone count configurable (3 to 6).
3. **Outside-in (WilMA):** all window seats first (A, F), then middle (B, E), then aisle (C, D). **Random within each wave.**
4. **Steffen:** for a plane with rows 1..30, board in this wave order, **back to front within each wave**:
   1. window seats on **even rows** (A and F, rows 30, 28, ..., 2)
   2. window seats on **odd rows** (rows 29, 27, ..., 1)
   3. middle seats on even rows
   4. middle seats on odd rows
   5. aisle seats on even rows
   6. aisle seats on odd rows

   Within a wave, alternate left-side and right-side passengers by row (so people in the same wave are spaced at least 2 rows apart). **Add a code comment noting that this follows Steffen (2008) and that the exact tie-breaking should be verified against the paper.**

All strategies must be unit-tested to produce a permutation of all 180 passengers with no duplicates.

---

## 8. Filipino Behavior Module (`behavior.py`)

Implemented as **independent, toggleable features**, each controlled in `config.yaml`. Apply behavior **after** the strategy produces the ideal order, by modifying the queue order and/or passenger attributes. Two named scenarios:

- **`baseline`**: all behavior features off (literature values only).
- **`philippine`**: features on with the defaults below.

| Feature | Model | Parameters (defaults, all `[ASSUMED]`) |
|---|---|---|
| **Group travel** | A fraction of passengers are assigned to groups of size 2-4 (draw sizes from weights 2: 50%, 3: 25%, 4: 25%), seated in **adjacent seats in the same row** where possible. The whole group boards together at the position of its **earliest-called member**, keeping internal order random. | `group_fraction = 0.40` (fraction of all passengers who belong to groups) |
| **Non-compliance** | Each passenger is compliant with probability `p`. Non-compliant passengers ignore their call and enter the queue at a **random position within the first 30% of the queue**. Group members share one compliance decision (the group moves together). | `compliance_rate = 0.70` |
| **Late passengers** | A small fraction is moved to the **end of the queue**. | `late_fraction = 0.03` |
| **Heavy hand-carry (pasalubong)** | Shift the bag-count distribution upward and lengthen stow time. | baseline bag probs: 0 bags 15%, 1 bag 60%, 2 bags 25%. Philippine: 0 bags 5%, 1 bag 45%, 2 bags 50% |
| **Bayanihan (bag help)** | When a passenger starts stowing, with probability `p_help` a **group mate (if in the same row) or the next passenger waiting directly behind** helps, cutting stow time by `help_reduction`. | `p_help = 0.15`, `help_reduction = 0.35` |
| **Seat-finding delay** | A fraction of passengers get extra delay at their row (first-time flyers, older passengers). | `seat_search_fraction = 0.10`, delay `Uniform(5, 15)` s |
| **Seat swapping** | After arriving at the row, a fraction of group members swap seats to sit together, adding extra shuffle time. **Implement last and keep optional.** | `swap_fraction = 0.05` |

**Design rules:** all behavior must consume randomness from the seeded RNG; each feature has an `enabled` flag; the module exposes `apply_behavior(order, passengers, rng, config) -> order`.

---

## 9. Parameters and Config

All values live in `config.yaml`. Mark the source of each value in a comment as `# source: [citation]` or `# ASSUMED`. **Do not invent citations.** If a value is not from a real paper, label it `# ASSUMED`. I will fill in the real citations.

Defaults (baseline scenario):

```yaml
cabin:
  rows: 30
  seat_letters: [A, B, C, D, E, F]
  row_pitch_m: 0.79            # ASSUMED (lit. range 0.75-0.81)
simulation:
  dt_seconds: 1
  entry_interval_s: 3          # ASSUMED
  base_seat_time_s: 2          # ASSUMED
  shuffle_time_s: {dist: uniform, low: 4, high: 10}   # ASSUMED
  max_ticks: 5000
passenger:
  walk_speed_mps: {dist: normal, mean: 0.8, sd: 0.15, min: 0.4}   # ASSUMED
  bags_probs: {0: 0.15, 1: 0.60, 2: 0.25}                         # ASSUMED
  stow_time_s:                  # lognormal per bag, ASSUMED
    zero_bags: 0
    per_bag: {dist: lognormal, mean: 9, sd: 4}
strategies:
  back_to_front_zones: 5
experiment:
  replications: 200
  base_seed: 12345
```

---

## 10. Experiments (`experiments.py`, `run_experiments.py`)

**Common random numbers:** for each replication `i`, generate the passenger population (bags, walk speeds, stow times, group assignment, compliance flags, late flags, delays) **once** using seed `base_seed + i`, then run **all four strategies on that same population**. Randomness inside strategies (within-zone shuffles) uses a separate RNG stream derived from the same seed so results stay reproducible.

**Experiment 1: Baseline comparison.** 4 strategies x 200 replications, `baseline` scenario.

**Experiment 2: Philippine scenario.** 4 strategies x 200 replications, `philippine` scenario.

**Experiment 3: Compliance sweep.** Philippine scenario with other features on; `compliance_rate` in `[1.0, 0.9, 0.75, 0.5, 0.25, 0.0]`.

**Experiment 4: Group sweep.** `group_fraction` in `[0.0, 0.2, 0.4, 0.6]`.

**Experiment 5: Sensitivity.** Vary mean stow time (x0.5, x1, x1.5, x2) and mean walk speed (0.6, 0.8, 1.0 m/s); report whether the strategy ranking changes.

**Experiment 6 (ablation):** turn each Filipino behavior on one at a time to show which contributes most to slowdown.

**Metrics per run (`metrics.py`):**
- `total_boarding_time_s`
- `aisle_interference_events`
- `seat_interference_events`
- `mean_wait_time_s` and `max_wait_time_s` per passenger
- strategy, scenario, seed, replication id, and all sweep parameters

Write raw per-run results to `results/*.csv` (tidy format, one row per run).

---

## 11. Statistics and Visualization

**`stats.py`:**
- mean, standard deviation, 95% confidence interval per strategy per condition (t-based)
- **paired t-test** between every strategy pair (pairing is valid because of common random numbers), with Bonferroni or Holm correction
- one-way repeated-measures or plain ANOVA as a supplement
- rank each strategy per condition and report rank changes

**Charts (matplotlib, PNG saved to `results/`, clear axis labels and titles, readable colors):**
1. Boxplot of total boarding time by strategy: baseline vs Philippine (side by side)
2. Bar chart of mean aisle and seat interference events by strategy
3. Line chart: mean boarding time vs compliance rate, one line per strategy, with CI bands
4. Line chart: mean boarding time vs group fraction, one line per strategy
5. Heatmap or table of strategy ranking under each sensitivity condition
6. Ablation bar chart: slowdown (%) caused by each behavior feature

**Animation (`animate.py`):** a matplotlib animation of one replication for a chosen strategy: the cabin as a grid with 30 rows, aisle in the middle, and colored dots for passengers by state (walking, blocked, stowing, seated). Show the time counter. Save as `.gif` or `.mp4` (`--strategy`, `--scenario`, `--seed` arguments). Make it viewable side by side for two strategies if feasible.

---

## 12. Milestones (build in this order; stop after each)

1. **M1: Skeleton + cabin + passenger + config loader.** Runs with no errors.
2. **M2: Core engine with random strategy, baseline only.** Prints total boarding time for one run.
3. **M3: Tests + invariants.** Single passenger, two passengers same row, hand-checkable numbers (see Section 13). All pass.
4. **M4: All four strategies + baseline experiment.** Produces Experiment 1 CSV and the boxplot.
5. **M5: Behavior module** (group, compliance, late, bags, seat search first; then bayanihan and seat swap). Experiment 2.
6. **M6: Experiments 3-6, stats, and all charts.**
7. **M7: Animation and `analysis.ipynb` + README.**

---

## 13. Verification and Validation Requirements

**Verification tests (pytest):**
- One passenger seated at row 1, seat C: time equals entry + stow + base seat time (hand-computed).
- One passenger at row 30: walking time is approximately `29 * pitch / speed`.
- Two passengers in the same row (A then C vs C then A): the C-then-A order must produce **zero** seat interference, and the A-then-C order must produce **one seat interference event** with added delay.
- A slow passenger stowing in row 5 blocks a faster passenger behind them heading to row 20; aisle interference counted exactly once.
- Each strategy returns a valid permutation.
- Invariants in Section 6 hold across 50 random full runs.
- Same seed twice gives identical results.

**Validation (baseline scenario) - qualitative agreement with published findings:**
- Random should be clearly faster than back-to-front, or at least not much worse. Back-to-front should be **no better than random**, and often worse.
- Outside-in and Steffen should be **the fastest**, both well below random.
- If results contradict these, print a warning in the baseline run and investigate before continuing. **Do not tune parameters to force the result without documenting it.**

---

## 14. Acceptance Criteria

- [ ] `pip install -r requirements.txt` then `python run_baseline.py` runs end to end and prints a summary table
- [ ] All tests pass with `pytest`
- [ ] Baseline ranking agrees with published qualitative findings (Section 13)
- [ ] Philippine scenario runs and shows measurable change vs baseline
- [ ] All 6 experiments produce CSVs and charts in `results/`
- [ ] Animation file generated
- [ ] Every parameter is in `config.yaml` with a `source` or `ASSUMED` comment
- [ ] `README.md` explains setup, how to run each experiment, and the model assumptions in plain language
- [ ] Code is commented and I can explain every module

---

## 15. Out of Scope

Pre-boarding for persons with disabilities, formal priority boarding tiers, multi-door or rear-door boarding, real airline data, gate changes and disruptions, 2D passenger physics or collision avoidance, and ATR 72 or other aircraft layouts (may be added later via config but not required).

---

## 16. Notes for the AI

- Prioritize **correctness and clarity** over cleverness.
- Keep the engine free of behavior-specific `if` clutter: behavior should modify passenger attributes and queue order **before** the engine runs, except for bayanihan and seat swapping, which hook into stowing and seating.
- If any rule in this document is ambiguous, pick the simplest reasonable interpretation, state it in a code comment and in the README, and continue.
- Do not fabricate literature values or citations. Label unknowns `ASSUMED`.
