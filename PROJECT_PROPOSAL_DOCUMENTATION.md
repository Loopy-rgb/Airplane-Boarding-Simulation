# Project Proposal & System Documentation: Modeling and Simulation

**Course:** Modeling and Simulation  
**Project:** Simulating and Comparing Airplane Boarding Strategies for Philippine Domestic Flights  
**Subject System:** Airbus A320 Single-Aisle Passenger Boarding (30 Rows, 180 Seats)  

---

## 1. Project Title
**Simulating and Comparing Airplane Boarding Strategies for Philippine Domestic Flights Using Agent-Based Modeling**

---

## 2. Introduction
Modeling and simulation (M&S) are essential tools used to represent complex real-world physical and human systems within a controlled computational environment. These tools enable optimization, predictive analysis, and operational decision-making without the prohibitive costs, physical risks, and operational disruptions of real-world experimentation.

In commercial aviation, passenger boarding is one of the primary operational bottlenecks in total aircraft turnaround time. Turnaround time directly dictates airline gate costs, fleet utilization rates, and on-time performance. While theoretical boarding strategies (such as Outside-In / WilMA and the Steffen Method) have been proposed in international operations research literature to reduce boarding times, their effectiveness under specific cultural and behavioral patterns—specifically **Philippine domestic flight operations** (characterized by frequent group travel, non-compliance with boarding calls, and heavy carry-on *pasalubong* luggage)—remains largely unstudied.

This project designs, implements, and evaluates an agent-based discrete time-step simulation model to analyze airplane boarding dynamics, comparing baseline literature conditions against Philippine domestic flight scenarios.

---

## 3. Problem Statement
Commercial airlines operating domestic routes in the Philippines frequently face boarding inefficiencies that lead to gate delays, passenger congestion, and elevated turnaround expenses. Traditional boarding strategies (notably Back-to-Front block boarding) are widely used by carriers, yet theoretical literature suggests they produce severe aisle bottlenecks.

Furthermore, existing boarding models assume idealized passenger behaviors (individual travel, strict compliance with boarding group announcements, and uniform carry-on luggage). There is a critical need to:
1. Model and evaluate how international boarding strategies perform when applied to an Airbus A320 configuration.
2. Quantify the performance degradation caused by real-world Filipino passenger behaviors:
   * **Group travel:** Families and companions clustering together and boarding out of order.
   * **Boarding non-compliance:** Passengers boarding ahead of their assigned group calls.
   * **Excess carry-on baggage (*pasalubong*):** Increased overhead bin competition and prolonged aisle stowage times.
   * **Seat interference:** In-row seat swapping and aisle shuffling.
3. Identify which boarding strategy offers the highest operational robustness for domestic Philippine airline operations.

---

## 4. Objectives

### General Objective
To develop, verify, and validate an agent-based discrete-time simulation framework to evaluate and compare airplane boarding strategies under ideal baseline conditions and realistic Philippine domestic passenger behaviors.

### Specific Objectives
1. **Develop Conceptual & Computational Models:**
   * Build a 1D cell-array cabin representation of an Airbus A320 (30 rows $\times$ 6 seats = 180 passenger capacity).
   * Implement discrete agent states (`QUEUED`, `WALKING`, `BLOCKED`, `STOWING`, `SEATING`, `SEATED`) and kinematic movement rules.
2. **Implement Four Boarding Strategies:**
   * **Random:** Baseline unorganized boarding.
   * **Back-to-Front (5 Zones):** Conventional industry block boarding.
   * **Outside-In (WilMA):** Window $\to$ Middle $\to$ Aisle wave boarding.
   * **Steffen Method:** Alternating-row, window-to-aisle parallelized boarding.
3. **Simulate Philippine Behavioral Scenarios:**
   * Incorporate modular, toggleable behavior features: Group Travel (2–4 members), Boarding Call Non-Compliance ($0\%\text{–}100\%$), Heavy Hand-Carry Luggage, Late Passengers (sent to end of queue), Bayanihan (luggage assistance), Seat-Finding Delays, and optional Seat Swapping (implemented but disabled by default).
4. **Conduct Sensitivity & Ablation Experiments:**
   * Execute 200 stochastic replications per condition using **Common Random Numbers (CRN)** for variance reduction.
   * Perform parameter sweeps on compliance rates ($1.0 \to 0.0$) and group fractions ($0.0 \to 0.6$).
5. **Validate & Verify Model Outputs:**
   * Execute automated unit and invariant testing suites (`pytest`) to guarantee zero cell collisions, monotonic forward motion, and seat uniqueness (invariant checks are active when `test_mode=True`).
   * Cross-validate baseline simulation **strategy rankings** against published empirical benchmarks (Steffen & Hotchkiss 2012). Note: absolute boarding times are not directly comparable because the Steffen & Hotchkiss experiment used a smaller 12-row mock cabin.
6. **Provide Data-Driven Airline Recommendations:**
   * Deliver actionable insights on strategy selection for Philippine domestic carriers (e.g., Cebu Pacific, Philippine Airlines, AirAsia Philippines).

---

## 5. Scope and Limitations of the Study

### Scope
* **Aircraft Configuration:** Airbus A320 single-aisle, single front door entry, 30 rows of 6 seats (3-3 layout: A, B, C \| D, E, F) totaling 180 seats at 100% load factor.
* **Strategies Evaluated:** Random, Back-to-Front (5 zones), Outside-In (WilMA), and Steffen Method.
* **Key Metrics Collected:** Total Boarding Time (seconds), Aisle Interference Events, Seat Interference Events, and Mean/Max Passenger Wait Times.
* **Interface & Tooling:** Automated CLI batch experiment runner, analytical Jupyter Notebook, and an interactive Flask/HTML5 2D visual dashboard.

### Limitations
* **Physical Assumptions:** Passenger movement is modeled as 1D discrete cell transitions ($1\text{ cell per row} = 0.79\text{ m}$ pitch) with a 1-second time tick ($dt = 1\text{ s}$).
* **Empirical Data Availability:** Due to the absence of public cabin CCTV telemetry from Philippine carriers, most simulation parameters are calibrated assumptions informed by boarding-simulation literature. Philippine cultural dynamics are evaluated as controlled sensitivity sweeps over hypothetical but plausible parameter ranges.
* **Aircraft Operations:** Boarding occurs strictly through a single front door (Door 1L); dual-door ramp boarding (front and rear stairs) is excluded from the primary scope.

---

## 6. Methodology

```
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│   1. DATA & PARAMETERS   │ ──> │  2. MODEL DEVELOPMENT   │ ──> │ 3. EXPERIMENTATION (CRN)│
│  • Lit. Distributions   │     │  • Agent State Machine  │     │  • 200 Replications/Exp │
│  • A320 Physical Specs  │     │  • 4 Strategy Permuters │     │  • Parameter Sweeps     │
│  • Config YAML Schema   │     │  • Behavior Modifiers   │     │  • Ablation Tests       │
└─────────────────────────┘     └─────────────────────────┘     └───────────┬─────────────┘
                                                                            │
┌─────────────────────────┐     ┌─────────────────────────┐                 │
│ 5. RECOMMENDATIONS &    │ <── │ 4. V&V & STATISTICAL    │ <───────────────┘
│    WEB DASHBOARD        │     │    ANALYSIS             │
│  • Final Report         │     │  • Pytest Invariants    │
│  • Interactive Visuals  │     │  • Paired t-test / ANOVA│
│  • Policy Guidance      │     │  • Literature Benchmarks│
└─────────────────────────┘     └─────────────────────────┘
```

### Phase 1: Data Collection & Parameter Calibration
All physical and stochastic parameters are decoupled into a centralized YAML configuration file (`config.yaml`). Parameters are labeled `ASSUMED` in the config unless a direct citation exists:
* **Row Pitch:** $0.79\text{ m}$ — **ASSUMED** (consistent with Airbus A320 standard 31-inch economy pitch; cite the specific Airbus AC page if verified).
* **Walking Speed:** $v \sim \mathcal{N}(0.8, 0.15^2)\text{ m/s}$, clipped at $v_{\min} = 0.4\text{ m/s}$. The $0.8\text{ m/s}$ mean is informed by Schultz (2018), who reports a fixed $0.8\text{ m/s}$ aisle speed; the Normal distribution form is an **ASSUMED** extension for inter-passenger variability.
* **Luggage Stowage Time:** Lognormal distribution per bag (mean $\approx 9\text{ s}$, SD $\approx 4\text{ s}$) — **ASSUMED** engineering default. (Schultz's (2018) empirical Weibull fit gives a mean of approximately $13.9\text{ s}$; we use the lower default, which is configurable.)
* **Seat Interference Delay:** $\mathcal{U}(4, 10)\text{ seconds}$ per blocking seated passenger — **ASSUMED**.

### Phase 2: Agent-Based Model Development
* **Cabin Model:** 1D array of 30 cells (front cell 0 to rear cell 29) + entry queue.
* **Passenger Agent:** Object tracking ID, assigned row/seat, luggage count, walk speed, stow time, current state, wait ticks, and interference counters.
* **Execution Rules:** Front-to-back priority cell updates each tick ($dt=1\text{s}$) preventing collisions.

### Phase 3: Experimentation & Common Random Numbers (CRN)
To eliminate random demographic noise between strategies, each replication seeds an identical passenger population before executing all four strategies.
* **Exp 1:** Baseline comparison (ideal conditions, 200 runs).
* **Exp 2:** Philippine scenario (all cultural modifiers active, 200 runs).
* **Exp 3:** Compliance rate sweep ($100\%, 90\%, 75\%, 50\%, 25\%, 0\%$).
* **Exp 4:** Group fraction sweep ($0\%, 20\%, 40\%, 60\%$).
* **Exp 5:** Baggage & walk speed sensitivity.
* **Exp 6:** Ablation breakdown of individual behavior components.

### Phase 4: Validation & Verification (V&V)
* **Automated Unit Tests:** 16 test cases in `pytest` verifying invariants (no shared cells, monotonic movement, seat uniqueness, exact permutation validity). Runtime invariant assertions are enabled via `test_mode=True`.
* **Literature Benchmark:** Strategy **rankings** are validated against Steffen & Hotchkiss (2012) physical mock-cabin trials. Absolute boarding times are not directly compared due to the smaller 12-row cabin used in that experiment.
* **Statistical Tests:**
  * **95% Bootstrap Confidence Intervals** for the mean difference between each pair of strategies per condition.
  * **Pairwise paired $t$-tests** between all $\binom{4}{2} = 6$ strategy pairs with plain **Bonferroni** $p$-value correction ($\alpha = 0.05$).
  * **Cohen's $d$** effect size for each pair.
  * **One-way ANOVA** (omnibus test; used as an approximation of repeated-measures ANOVA).
  * **Mann-Whitney U tests** (non-parametric fallback) for baseline vs. Philippine pairwise comparisons.

---

## 7. Expected Outputs
1. **Fully Runnable Simulation Engine:** Modular Python package (`boarding_sim/`).
2. **Automated Verification Suite:** 16 automated invariant unit tests in `pytest`.
3. **Empirical CSV Datasets:** Raw per-run simulation records (`exp1_baseline.csv` through `exp6_ablation.csv`).
4. **Publication-Quality Visualizations:**
   * Baseline vs. Philippine boarding time boxplots.
   * Aisle vs. seat interference event bar charts.
   * Compliance and group sweep response curves.
   * Feature ablation slowdown charts.
5. **Interactive Web Dashboard:** Live browser-based visual simulation interface (`http://localhost:5050`) with real-time 2D cabin animations and parameter controls.
6. **Comprehensive Analytical Report:** Written documentation with statistical tables, confidence intervals, and airline policy recommendations.

---

## 8. Tools and Technologies

| Category | Tools / Libraries Used | Purpose |
|---|---|---|
| **Programming Language** | Python 3.10+ | Core object-oriented simulation engine |
| **Scientific Computing** | `numpy`, `pandas`, `scipy` | Random number generation, data handling, paired $t$-tests, ANOVA |
| **Data Visualization** | `matplotlib` | Generation of publication-grade figures and boxplots |
| **Testing Framework** | `pytest` | Automated verification of physical invariants and kinematics |
| **Configuration Management**| `PyYAML` | Centralized parameter storage (`config.yaml`) |
| **Web Dashboard** | Flask, HTML5 Canvas, Vanilla CSS, JS | Real-time interactive 2D passenger boarding visualizer |

---

## 9. Project Timeline & Milestones

| Milestone | Target Duration | Key Activities & Deliverables |
|---|:---:|---|
| **M1: Foundation & Cabin Architecture** | Week 1 | Configuration schema (`config.yaml`), `Cabin` seat layout, `Passenger` dataclass, and single-passenger kinematic unit tests. |
| **M2: Simulation Engine & Strategies** | Week 2 | Time-step engine loop, cell transition logic, collision prevention, implementation of 4 boarding strategies, and invariant tests. |
| **M3: Filipino Behavior Module** | Week 3 | Implementation of group travel, non-compliance, late passengers, heavy baggage distributions, bayanihan luggage help, and seat-search delays. |
| **M4: Experimentation & Statistical Suite**| Week 4 | Execution of Experiments 1–6 (200 replications each with CRN), CSV generation, paired $t$-tests, and chart rendering. |
| **M5: Web Dashboard & Final Report** | Week 5 | Interactive Flask web dashboard, Jupyter analysis notebook, model validation documentation, and final report. |

---

## 10. Project Guidelines & Roles
* **Collaboration Standards:** Modular repository structure adhering to clean code standards, typed Python signatures, and deterministic RNG seeding.
* **Code Integrity:** Strictly decoupled configuration without hardcoded constants; physical invariant assertions are enforced every tick when running in `test_mode=True` (used by the automated test suite; disabled in normal experiment runs for performance).
* **Deliverables:** Working repository, automated test suite, interactive dashboard, generated figures, and documentation report.

---

## 11. Conclusion & Significance
This project demonstrates the practical power of Modeling and Simulation in solving high-impact transportation logistics problems. By integrating empirical human behavioral modifiers (group dynamics, non-compliance, baggage friction) into classical boarding algorithms, this study bridges the gap between theoretical operations research and real-world aviation operations in the Philippines. 

The resulting framework provides Philippine airline operators with an objective, data-backed foundation to optimize boarding calls, improve turnaround punctuality, and enhance the domestic passenger experience.

---

## 12. References
1. **Airbus S.A.S.** (2024). *Airbus A320 Aircraft Characteristics: Airport and Maintenance Planning* (June 2024 ed.). Airbus Technical Publications. Retrieved from https://aircraft.airbus.com/en/customer-care/fleet-wide-care/airport-operations-and-aircraft-characteristics/aircraft-characteristics
2. **Sargent, R. G.** (2013). *Verification and validation of simulation models*. Journal of Simulation, 7(1), 12–24.
3. **Schultz, M.** (2018). *Field trial measurements to validate a stochastic aircraft boarding model*. Aerospace, 5(1), 27. https://doi.org/10.3390/aerospace5010027
4. **Schultz, M.** (2018). *Implementation and application of a stochastic aircraft boarding model*. Transportation Research Part C: Emerging Technologies, 90, 334–349. https://doi.org/10.1016/j.trc.2018.03.016
5. **Steffen, J. H.** (2008). *Optimal boarding method for airline passengers*. Journal of Air Transport Management, 14(3), 146–150.
6. **Steffen, J. H., & Hotchkiss, J.** (2012). *Experimental test of airplane boarding methods*. Journal of Air Transport Management, 18(1), 64–67. https://doi.org/10.1016/j.jairtraman.2011.10.003
7. **Van den Briel, M. H. L., Villalobos, J. R., Hogg, G. L., Lindemann, T., & Mulé, A. V.** (2005). *America West Airlines develops efficient boarding strategies*. Interfaces, 35(3), 191–201.
