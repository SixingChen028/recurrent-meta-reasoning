# Learning to Select Computations in Recurrent Neural Circuits

**Sixing Chen\*, Frederick Callaway\*, Sreejan Kumar, Shira M. Lupkin, Joni D. Wallis, Vincent B. McGinty, Erin L. Rich, and Marcelo G. Mattar**

*\* equal contribution*

Link to the paper: https://www.biorxiv.org/content/your-link-here

---

## Overview

This repository contains code for the paper *"Learning to select computations in recurrent neural circuits"* (Chen, Callaway et al., 2026).

---

## Repository Structure

Each directory corresponds to one of the four case studies in the paper:

| Directory          | Case Study                                                   |
| ------------------ | ------------------------------------------------------------ |
| `eyechoice/`       | Gaze-based value sampling in a simple choice task (Callaway et al., 2021) |
| `richwallis/`      | Neural dynamics in macaque OFC in a binary choice task (Rich & Wallis, 2016) |
| `mcgintylupkin/`   | Neural geometry in macaque OFC in a binary choice task  (McGinty & Lupkin, 2023) |
| `eyeplan/`         | Gaze-based planning strategy in a planning task  (Callaway et al., 2024) |
| `vikbladhburgess/` | Step-by-step mental simulation in human MEG in a planning task (Vikbladh et al., 2024) |

Each directory follows the same structure:

```
<case_study>/
├── modules/
│   ├── network.py        # Recurrent neural network with actor-critic architecture
│   ├── environment.py    # Task environment
│   ├── a2c.py            # REINFORCE with baseline trainer
│   ├── simulation.py     # Evaluation utilities
│   ├── argument.py       # Hyperparameter settings
│   ├── replaybuffer.py   # Episode replay buffer for trainer
│   └── utils.py          # Utilities for the case study
├── train.py              # Training
├── simulate*.py          # Simulation / evaluation
├── analysis_*.py         # Analysis
├── plot_*.py             # Plotting
├── run_*.sh              # SLURM cluster script for running python code
└── submit_*.sh           # SLURM cluster script for job submission
```

---

## Getting Started

### Requirements

```bash
pip install torch gymnasium numpy scipy scikit-learn matplotlib seaborn pandas networkx statsmodels
```

> Python 3.9+ and PyTorch 2.0+ are recommended.

### Training

Each case study can be trained independently. Training each agent requires running 5 random seeds and can take several hours; **a CPU cluster is strongly recommended**.

Each case study directory contains two sets of shell scripts:

- `run_*.sh` — SLURM job scripts (resource requests, array indices, Python invocation)
- `submit_*.sh` — wrapper scripts that loop over hyperparameter grids and call `sbatch run_*.sh`

To launch all training jobs for a case study, simply run the submit script:

```bash
# Simple choice task
cd eyechoice
sbatch submit_train.sh

# Planning task
cd eyeplanbash
sbatch submit_train.sh

# All other case studies follow the same pattern
cd <case_study>
sbatsh submit_train.sh
```

Each `run_train.sh` requests 8 CPUs, 10 GB RAM per CPU, and runs a SLURM array of 5 jobs (one per random seed). Typical wall-clock times:

| Case Study        | Time per seed |
| ----------------- | ------------- |
| `eyechoice`       | ~4 hours      |
| `richwallis`      | ~4 hours      |
| `mcgintylupkin`   | ~4 hours      |
| `eyeplan`         | ~12 hours     |
| `vikbladhburgess` | ~4 hours      |

Results (model checkpoints and logs) are saved under `<case_study>/results/`.

### Simulation & Analysis

After training, run simulation to generate evaluation data, then run analysis and plotting scripts:

```bash
sbatch run_simulate_*.sh
sbatch run_analysis_*.sh
sbatch run_plot_*.sh
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{chen2026learning,
  title   = {Learning to select computations in recurrent neural circuits},
  author  = {Chen, Sixing and Callaway, Frederick and Kumar, Sreejan and Lupkin, Shira M. and Wallis, Joni D. and McGinty, Vincent B. and Rich, Erin L. and Mattar, Marcelo G.},
  year    = {2026},
}
```

---

## Contact

Questions? Please open an issue or contact Sixing Chen at sixing.chen@nyu.edu.