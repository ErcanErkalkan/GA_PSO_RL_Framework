# DLGPR Framework: Dynamic Layered GA-PSO-RL for Real-Time Game AI

This repository contains the reproducibility package and source code for the paper:  
**"A Dynamic Layered GA--PSO--RL Framework for Real-Time Game AI Under Compute Budgets"**

## Overview

The DLGPR framework introduces a budget-aware hybrid optimization architecture designed for games with strict real-time constraints (hard wall-clock budgets). It dynamically allocates computational resources between:

* **Exploration:** Genetic Algorithms (GA) with Novelty Search.
* **Exploitation:** Particle Swarm Optimization (PSO).
* **Adaptation:** Reinforcement Learning (PPO-lite) with Entropy Regularization.

The system uses a unified candidate pool and handshake mechanisms (Injection & Distillation) to bridge evolutionary and gradient-based methods.

## Directory Structure

* `src/`: Contains the main Python implementation (`main.py`).
* `paper/`: Contains the IEEE LaTeX source code for the manuscript.
* `out/`: (Generated at runtime) Stores logs, plots, and tables.

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/KULLANICI_ADINIZ/DLGPR-Framework.git](https://github.com/KULLANICI_ADINIZ/DLGPR-Framework.git)
   cd DLGPR-Framework

2. Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:

pip install -r requirements.txt

## Usage
To run the complete experiment suite (DLGPR + Baselines) with default settings:

python src/main.py --run_name experiment_01 --intervals 120

## Command Line Arguments
--env: Gym environment ID (e.g., CartPole-v1). Defaults to internal ToyPOMDP if not provided.

--B_tau_ms: Hard wall-clock budget per interval in milliseconds (default: 120.0).

--seeds: Number of random seeds to run (default: 5).

--methods: Comma-separated list of methods to run (e.g., DLGPR,GA-only,RL-only).

## Outputs
After a successful run, the out/ directory will contain:

Logs: CSV files with per-interval scheduler decisions.

Figures: Latency traces, allocation plots, and conceptual paper figures.

Tables: mainresults.tex (IEEE format) and mainresults.csv.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
If you use this code in your research, please cite:

@article{dlgpr2025,
  title={A Dynamic Layered GA--PSO--RL Framework for Real-Time Game AI Under Compute Budgets},
  author={Erkalkan, Ercan},
  journal={},
  year={2025}
}