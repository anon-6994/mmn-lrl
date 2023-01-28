# Lifelong Reinforcement Learning with Modulating Masks
Implementation of modulatory mask combined with PPO. The repository contains MASK RI/LC/BLC implementations. Please see EWC branch for implemenation of PPO and Online EWC.

**This branch contains the code to run MASK RI_D (discrete-value modulatory mask) for Continual World experiment.**

The code was developed on top of the existing [DeepRL](https://github.com/ShangtongZhang/DeepRL) repository, extending PPO RL agents to lifelong learning setting.

Python 3.9 and PyTorch 1.12.0 were used for experiments in the paper.

For Mask RI/LC/BLC experiment and other baselines in the Procgen benchmark, please visit this [repository](#).

## Evaluation environments
- [CT-graph](https://github.com/soltoggio/CT-graph)
- [Minigrid](https://github.com/Farama-Foundation/gym-minigrid)
- [Continual World](https://github.com/awarelab/continual_world) (see note below)

## Requirements
- See requirements.txt file
- See [CT-graph](https://github.com/soltoggio/CT-graph) requirements.
- See [Minigrid](https://github.com/Farama-Foundation/gym-minigrid) requirements.
- See [Continual World](https://github.com/awarelab/continual_world) requirements and how to install. Note, MuJoCo (now freely available) is required to run Continual World

## Usage
The branch is specific for running experiments for discrete/binary masks for MASK RI agents in continualworld's CW10 curriculum. See sample command below:
```
# MASK RI_D agent in CW10
python train_continualworld.py ll_supermask --new_task_mask random --seed 436
```

Note: 
- the full list of commands to run experiments in the paper can be found in the `paper_experiments.txt` file.
- in the `master` branch, the above command will execute as a MASK RI\_C (random initialisation of mask agent that uses continuous value masks) experiment.

## Note on Continual World
The Continual World benchmark was built on top of the [Meta-World](https://github.com/rlworkgroup/metaworld) benchmark, which comprise of a number of simulated robotics tasks. The originally released Continual World employed the use of version 1 (v1) Meta-World environments. However, the Meta-World v1 environments contained some issues in the reward function (discussed [here](https://github.com/rlworkgroup/metaworld/issues/226) and [here](https://github.com/awarelab/continual_world/issues/2)) which was fixed in the updated v2 environments. Therefore, the experiments in the paper employed the use of the v2 environment for each task in the Continual World.
