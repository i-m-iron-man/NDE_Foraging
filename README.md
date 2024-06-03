# NDE Foraging
This repository supplements our paper ["A Dynamical Systems Approach to Optimal Foraging"](https://www.biorxiv.org/content/10.1101/2024.01.20.576399v1)</br>
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)</br>
![Alt Text](https://github.com/i-m-iron-man/NDE_Foraging/blob/main/media/foraging.gif)


## Overview
This repository contains code for the 3 experiments presented in the paper. It is developed using the JAX ecosystem</br>
- Experiment 1 highlights the learning process involved in adaptive patch foraging and analysis of the learned agent at various stages of training. </br>
- Experiment 2 presents an analysis of evidence accumulation mechanism observed in the adaptive agent. </br>
- Experiment 3 presents an analysis of change in average patch residing time of the adaptive agent with respect to change in growth rate of the resources in the environment.</br>
- Experiment 4 shows how the setup can be extended to nosiy setup like noise in observations.
- Experiment 5 shows how the setup can be extended to nosiy setup like multi-agent setups.

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
- [source](https://github.com/i-m-iron-man/NDE_Foraging/tree/main/source)
    - [exp_1_basic_foraging](https://github.com/i-m-iron-man/NDE_Foraging/tree/main/source/exp_1_basic_foraging): directory for Experiment 1.
        - [main.py](https://github.com/i-m-iron-man/NDE_Foraging/blob/main/source/exp_1_basic_foraging/main.py): A Python script that executes the learning process for different seeds.
        - [plot_values.ipynb](https://github.com/i-m-iron-man/NDE_Foraging/blob/main/source/exp_1_basic_foraging/plot_values.ipynb): An IPython notebook to plot the loss curve.
        - [analysis.ipynb](https://github.com/i-m-iron-man/NDE_Foraging/blob/main/source/exp_1_basic_foraging/analysis.ipynb): An IPython notebook to observe the behaviour of agent at various stages of training.
        - [render.py](https://github.com/i-m-iron-man/NDE_Foraging/blob/main/source/exp_1_basic_foraging/render.py): A Python Script to render the learned agent.
    - [exp_2_evidence_accumulation](https://github.com/i-m-iron-man/NDE_Foraging/tree/main/source/exp_2_evidence_accumulation): directory for Experiment 2.
        - [evidence.ipynb](https://github.com/i-m-iron-man/NDE_Foraging/blob/main/source/exp_2_evidence_accumulation/evidence.ipynb): An IPython notebook to analyse the neuronal dynamics of the learned agent.
    - [exp_3_diff_growth_rate](https://github.com/i-m-iron-man/NDE_Foraging/tree/main/source/exp_3_diff_growth_rate): directory for Experiment 3.
        - [growth_rate_08f.py](https://github.com/i-m-iron-man/NDE_Foraging/blob/main/source/exp_3_diff_growth_rate/growth_rate_08f.py): A Python file for learning in the environment having growth rate of 0.08.
        - [growth_rate_10f.py](https://github.com/i-m-iron-man/NDE_Foraging/blob/main/source/exp_3_diff_growth_rate/growth_rate_10f.py): A Python file for learning in the environment having growth rate of 0.1.
        - [growth_rate_12f.py](https://github.com/i-m-iron-man/NDE_Foraging/blob/main/source/exp_3_diff_growth_rate/growth_rate_12f.py): A Python file for learning in the environment having growth rate of 0.12.
        - [growth_rate_time_analysis.ipynb](https://github.com/i-m-iron-man/NDE_Foraging/blob/main/source/exp_3_diff_growth_rate/growth_rate_time_analysis.ipynb): An IPython notebook to sample and compare average patch residing time for agent trajectories in the 3 environemts.
        - [avg_growth_rate_plt.ipynb](https://github.com/i-m-iron-man/NDE_Foraging/blob/main/source/exp_3_diff_growth_rate/avg_growth_rate_plt.ipynb): An IPython notebook to plot the average patch residing time across the 3 environments for the first 3 patch visits.
   - [exp_4_noise](https://github.com/i-m-iron-man/NDE_Foraging/tree/main/source/exp_4_noise): directory for Experiment 4.
    	- [min.py](https://github.com/i-m-iron-man/NDE_Foraging/blob/main/source/exp_4_noise/main.py): A python file that executes learning process for noisy setup for different seeds.
      	- [render.py](https://github.com/i-m-iron-man/NDE_Foraging/blob/main/source/exp_4_noise/render.py): A Python script to render the learned agents.
      	- [plot.ipynb](https://github.com/i-m-iron-man/NDE_Foraging/blob/main/source/exp_4_noise/plot.ipynb): An IPython notebook to plot different performance criteria for comparison.
   - [exp_5_multi_Agent](https://github.com/i-m-iron-man/NDE_Foraging/tree/main/source/exp_5_multi_agent): directory for Experiment 5.
        - [main_g_01.py](https://github.com/i-m-iron-man/NDE_Foraging/blob/main/source/exp_5_multi_agent/main_g_01.py): A Python script for learning in an environemt where growth rate of both resources is 0.1.
        - [main_g_03.py](https://github.com/i-m-iron-man/NDE_Foraging/blob/main/source/exp_5_multi_agent/main_g_03.py): A Python script for learning in an environemt where growth rate of the first resource is 0.1 and second resource is 0.3
        - [main_g_10.py](https://github.com/i-m-iron-man/NDE_Foraging/blob/main/source/exp_5_multi_agent/main_g_10.py): A Python script for learning in an environemt where growth rate of the first resource is 0.1 and second resource is 1.0
        - [render.py](https://github.com/i-m-iron-man/NDE_Foraging/blob/main/source/exp_5_multi_agent/render.py): A Python script for rendering the multi-agent setup
        - [infer.ipynb](https://github.com/i-m-iron-man/NDE_Foraging/blob/main/source/exp_5_multi_agent/infer.ipynb): An IPython notebook to plot relevant trends of the experiment.
      
## Acknowledgements
This software is part of the project Dutch Brain Interface Initiative (DBI<sup>2</sup>) with project number 024.005.022 of the research programme Gravitation which is (partly) financed by the Dutch Research Council (NWO).

## Citation
For citing this work you can cite the paper

```
@article {Chaturvedi2024.01.20.576399,
	author = {Siddharth Chaturvedi and Ahmed ElGazzar and Marcel van Gerven},
	title = {A Dynamical Systems Approach to Optimal Foraging},
	elocation-id = {2024.01.20.576399},
	year = {2024},
	doi = {10.1101/2024.01.20.576399},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/01/22/2024.01.20.576399},
	eprint = {https://www.biorxiv.org/content/early/2024/01/22/2024.01.20.576399.full.pdf},
	journal = {bioRxiv}
}
```
