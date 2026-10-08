# Simple Testing & Validation Guide
**A quick, step-by-step guide to test and verify the simulation yourself.**

---

## 🚀 Quick Start: 3 Ways to Test the Simulation

---

### Method 1: Interactive Web Dashboard (Visual Testing)

The easiest way to test the simulation is through the visual dashboard in your browser.

1. **Start the Web App** (if not already running):
   Open your terminal inside the project folder and run:
   ```bash
   cd boarding_sim
   python3 web_app.py
   ```
2. **Open the Dashboard in your browser:**
   Go to: **[http://localhost:5050](http://localhost:5050)**

3. **What to test and click:**
   * **Test Strategy Rankings:**
     * Select **Steffen** → click **Run Simulation** → check the total seconds (typically ~634–756 s, fastest).
     * Select **Back to Front** → click **Run Simulation** → notice it takes much longer (~1342–1859 s, slowest).
     * Select **Outside In (WilMA)** → notice window seats fill first, then middle, then aisle.
   * **Test Filipino Behavior:**
     * Turn ON **Group Travel** slider (e.g. 40%) → watch passengers in the same group board together and sit side-by-side.
     * Turn ON **Heavy Hand-Carry (Pasalubong)** → notice stow times increase and aisle queues back up.
     * Lower **Compliance Rate** to 30% → watch passengers ignore their boarding group and jump in line.

---

### Method 2: Automated Unit Tests (Check if Code Works)

Run the built-in automated test suite to make sure all movement rules and seat rules work properly.

1. **Run the tests in terminal:**
   ```bash
   cd boarding_sim
   python3 -m pytest -v
   ```

2. **What you should see:**
   You should see 16 green `PASSED` checks:
   ```
   tests/test_invariants.py::test_invariants_hold_across_50_runs PASSED
   tests/test_invariants.py::test_same_seed_identical_results PASSED
   tests/test_single_passenger.py::test_single_passenger_row1_seat_c PASSED
   tests/test_strategies.py::test_outside_in_window_before_aisle PASSED
   tests/test_two_same_row.py::test_order_C_then_A_has_one_seat_interference PASSED
   ...
   ============================== 16 passed in 3.26s ==============================
   ```

---

### Method 3: Run Baseline Experiments (Generate CSVs & Plots)

To run the scientific benchmark comparison across all 4 strategies (200 flights each):

1. **Run in terminal:**
   ```bash
   cd boarding_sim
   python3 run_baseline.py
   ```
2. **Where to see the results:**
   * Look inside the folder: `boarding_sim/results/`
   * You will see summary tables and generated comparison charts (`.png` files) comparing all 4 boarding strategies.

---

## 📋 Simple Validation Checklist (Sanity Checks)

How do you know the simulation is giving valid, realistic numbers? Check these 4 simple rules:

| # | Check | Expected Result | Why? |
|---|---|---|---|
| **1** | **Strategy Order** | Steffen is fastest (~634–756 s), Back-to-Front is slowest (~1342–1859 s). | Steffen allows 6+ people to stow luggage at once across the plane. Back-to-front causes everyone to jam in the same rear rows. |
| **2** | **Seat Interference** | Outside-In (WilMA) has almost 0 seat delays. Random has lots of seat delays. | In WilMA, window seats sit before middle and aisle seats, so nobody has to get up to let someone in. |
| **3** | **Heavy Luggage Effect** | Turning on *Heavy Hand-Carry* increases total time by ~5–14% depending on strategy (smallest for Steffen, largest for Back-to-Front). | More bags take longer to stow, but Steffen's row spacing absorbs stow conflicts better than bunched strategies. |
| **4** | **Repeatability (Seed)** | Running with the same seed (e.g., `seed=12345`) gives the exact same result every time. | Validates that the simulation is scientifically reproducible. |

---

## 🔍 How to Test Single Scenarios Quickly via Python

If you want to test one specific passenger or single flight in a python shell:

```python
from boarding import load_config
from boarding.cabin import Cabin
from boarding.simulation import generate_passengers, run_simulation
from boarding.strategies import strategy_outside_in
import numpy as np

# 1. Load default config
config = load_config("config.yaml")

# 2. Setup cabin and passengers
cabin = Cabin(rows=30)
rng = np.random.default_rng(12345)
passengers = generate_passengers(cabin, config, rng)

# 3. Apply a strategy (e.g. Outside-In)
queue = strategy_outside_in(passengers, rng)

# 4. Run the simulation
result = run_simulation(cabin, queue, config, rng)

print(f"Boarding finished in: {result.total_boarding_time_s} seconds ({result.total_boarding_time_s / 60:.1f} minutes)")
print(f"Aisle blocked events: {result.aisle_interference_events}")
print(f"Seat shuffle events:  {result.seat_interference_events}")
```
