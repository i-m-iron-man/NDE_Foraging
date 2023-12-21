# NDE Foraging
This repository supplements our paper "A Dynamical Systems Approach to Optimal Foraging"</br>
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository contains code for the 3 experiments presented in the paper. </br>
- Experiment 1 highlights the learning process involved in adaptive patch foraging and analysis of the learned agent at various stages of training. </br>
- Experiment 2 presents an analysis of evidence accumulation mechanism observed in the adaptive agent. </br>
- Experiment 3 presents an analysis of change in average patch residing time of the adaptive agent with respect to change in growth rate of the resources in the environment.</br>

## Requirements
In order to run the code following requirements must be satisfied. </br>
- [Python](https://www.python.org/downloads/) - 3.10.10
- [IPython](https://ipython.org/) - 8.10.0
- [JAX](https://jax.readthedocs.io/en/latest/installation.html) - 0.4.13
- [Diffrax](https://github.com/patrick-kidger/diffrax) - 0.4.0
- [Equinox](https://github.com/patrick-kidger/equinox) - 0.10.6
- [Optax](https://github.com/google-deepmind/optax) - 0.1.4
- [NumPy](https://numpy.org/install/) - 1.26.1
- [Matplotlib](https://matplotlib.org/) - 3.6.3

## File structure
This repository is stuctured as follows
- [source]()
    - [exp_1_basic_foraging](): directory for Experiment 1.
        - [main.py](): A Python script that executes the learning process for different seeds.
        - [plot_values,ipynb](): An IPython notebook to plot the loss curve.
        - [analysis,ipynb](): An IPython notebook to observe the behaviour of agent at various stages of training.
        - [render.py](): A Python Script to render the learned agent.
    - [exp_2_evidence_accumulation](): directory for Experiment 2.
        - [evidence.ipynb](): An IPython notebook to analyse the neuronal dynamics of the learned agent.
    - [exp_3_diff_growth_rate](): directory for Experiment 3.
        - [growth_rate_08f.py](): A Python file for learning in the environment having growth rate of 0.08.
        - [growth_rate_10f.py](): A Python file for learning in the environment having growth rate of 0.1.
        - [growth_rate_12f.py](): A Python file for learning in the environment having growth rate of 0.12.
        - [growth_rate_time_analysis.ipynb](): An IPython notebook to sample and compare average patch residing time for agent trajectories in the 3 environemts.
        - [avg_growth_rate_plt.ipynb](): An IPython notebook to plot the average patch residing time across the 3 environments for the first 3 patch visits.

      
      
## Acknowledgements
This software is part of the project Dutch Brain Interface Initiative (DBI$^2$) with project number 024.005.022 of the research programme Gravitation which is (partly) financed by the Dutch Research Council (NWO).

## Cite
