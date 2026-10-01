[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Solving Markov Decision Processes via Largest-size Average Estimator

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

This repository contains the code and numerical results for the paper
[Solving Markov Decision Processes via Largest-size Average Estimator](https://doi.org/10.1287/ijoc.2024.0863)
by Peiwen Yu, Qidong Lai, Yuan Tian, and Guangwu Liu.

## Cite

To cite the contents of this repository, please cite both the paper and
this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2024.0863

https://doi.org/10.1287/ijoc.2024.0863.cd

Below is the BibTeX for citing this snapshot of the repository.

```bibtex
@misc{Yu2026LSA,
  author =        {Peiwen Yu and Qidong Lai and Yuan Tian and Guangwu Liu},
  publisher =     {INFORMS Journal on Computing},
  title =         {{Solving Markov Decision Processes via Largest-size Average Estimator}},
  year =          {2026},
  doi =           {10.1287/ijoc.2024.0863.cd},
  url =           {https://github.com/INFORMSJoC/2024.0863},
  note =          {Available for download at https://github.com/INFORMSJoC/2024.0863},
}
```

## Repository structure

```text
.
├── README.md
├── AUTHORS
├── LICENSE
├── requirements.txt
├── scripts/
│   ├── value_estimation/
│   │   ├── intruder_monitor.py
│   │   ├── inventory_control.py
│   │   ├── ttt3x3_value_estimation.py
│   │   └── ttt4x4_value_estimation.py
│   └── policy_performance/
│       ├── ttt4x4_random.py
│       └── ttt4x4_softmax.py
└── results/
    ├── value_estimation/
    │   ├── intruder_monitor.csv
    │   ├── inventory_control.csv
    │   ├── ttt3x3.csv
    │   └── ttt4x4.csv
    └── policy_performance/
        ├── ttt4x4_random.csv
        └── ttt4x4_softmax.csv
```



## Requirements

The experiments are implemented in Python. Install the required packages
with:

```bash
pip install -r requirements.txt
```

The main dependencies are NumPy, Numba, and tqdm. Numba is used to
accelerate computationally intensive experiments.

## Reproducing the experiments

Run the commands below from the repository root.

### Value-estimation experiments



#### Intruder monitoring

```bash
python scripts/value_estimation/intruder_monitor.py
```

The script reproduces the main and robustness experiments for the
intruder-monitoring problem. The optimal benchmark values are computed
by dynamic programming. Output is saved to
`results/value_estimation/intruder_monitor.csv`.

#### Inventory control

```bash
python scripts/value_estimation/inventory_control.py
```

The script reproduces the main and robustness experiments for the
inventory-control problem. The optimal benchmark values are computed by
dynamic programming. Output is saved to
`results/value_estimation/inventory_control.csv`.

#### 3x3 Tic-Tac-Toe value estimation

```bash
python scripts/value_estimation/ttt3x3_value_estimation.py
```

The script reproduces the 3x3 Tic-Tac-Toe value-estimation experiment.
The exact benchmark is computed by expectimax. Output is saved to
`results/value_estimation/ttt3x3.csv`.

#### 4x4 Tic-Tac-Toe value estimation

```bash
python scripts/value_estimation/ttt4x4_value_estimation.py
```

The script reproduces the 4x4 Tic-Tac-Toe value-estimation experiment.
The exact benchmark is computed by expectimax. Output is saved to
`results/value_estimation/ttt4x4.csv`.

### Policy-performance experiments

For each policy model, the AMS estimates are used to construct the
Player-2 policy, which is then evaluated by simulation.

#### 4x4 Tic-Tac-Toe with a uniform-random Player-1 policy

```bash
python scripts/policy_performance/ttt4x4_random.py
```

Output is saved to `results/policy_performance/ttt4x4_random.csv`.

#### 4x4 Tic-Tac-Toe with a softmax Player-1 policy

```bash
python scripts/policy_performance/ttt4x4_softmax.py
```

Output is saved to `results/policy_performance/ttt4x4_softmax.csv`.

## Implementation and reproducibility

For the value-estimation experiments, different estimators use the same set of replication seeds within each experimental setting and sample size to facilitate paired and reproducible comparisons. Each script contains the complete configuration for its experiment, including the initial state where applicable, sample sizes, recursion depth or horizon, number of replications, and other experiment-specific parameters.

The CSV files under `results/` contain the numerical results generated
for the computational experiments. Re-running a script writes its output
to the corresponding CSV file listed above.

## License

This repository is released under the MIT License. See `LICENSE` for
details.
