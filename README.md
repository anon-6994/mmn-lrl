# Lifelong Reinforcement Learning with Modulating Masks
Implementation of modulatory mask combined with PPO. The repository contains MASK RI/LC/BLC implementations. Please see EWC branch for implemenation of PPO and Online EWC.

The code was developed on top of the existing [DeepRL](https://github.com/ShangtongZhang/DeepRL) repository, extending PPO RL agents to lifelong learning setting.

Python 3.9 and PyTorch 1.12.0 were used for experiments in the paper.

For Mask RI/LC/BLC experiments and other baselines in the Procgen benchmark, please visit this [repository](#).

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
Example commands below using [Minigrid](https://github.com/Farama-Foundation/gym-minigrid) environment.
To run agents in the minigrid (MG10) curriculum defined in the paper, use the command below:

```
# baseline PPO agent.
python train_minigrid.py baseline --seed 86

# random initialization of mask per task (MASK RI) agent.
python train_minigrid.py ll_supermask --new_task_mask random --seed 86

# linear combination of mask (MASK LC) agent.
python train_minigrid.py ll_supermask --new_task_mask linear_comb --seed 86
```

Note: 
- the command to run a balanced linear combination (MASK BLC) agent is the same as the MASK LC command above, but should be run in the `exp_maskblc` git branch.
- the full list of commands to run experiments in the paper can be found in the `paper_experiments.txt` file.
- sample commands and the full list of commands for `ewc` experiments in the paper can be found in the `exp_ewc` git branch. 
- sample commands and the full list of commands for setting up the single task expert (STE) experiments can be found in the `exp_ste` git branch.
- In the continualworld curriculum (CW10), the random initialization mask agent implemented in this branch is the MASK RI\_C (continuous values mask). The sample command to run MASK RI_\D in CW10 can be found in the `exp_maskri_discrete_mask_cw10` git branch.

## Note on Continual World
The Continual World benchmark was built on top of the [Meta-World](https://github.com/rlworkgroup/metaworld) benchmark, which comprise of a number of simulated robotics tasks. The originally released Continual World employed the use of version 1 (v1) Meta-World environments. However, the Meta-World v1 environments contained some issues in the reward function (discussed [here](https://github.com/rlworkgroup/metaworld/issues/226) and [here](https://github.com/awarelab/continual_world/issues/2)) which was fixed in the updated v2 environments. Therefore, the experiments in the paper employed the use of the v2 environment for each task in the Continual World.
