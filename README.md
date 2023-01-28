# Lifelong Reinforcement Learning with Modulating Masks
Implementation of modulatory mask combined with PPO. The repository contains MASK RI/LC/BLC implementations.

**This branch contains the implementation of PPO combined with Online EWC.**

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
EWC branch
Example commands below using [Minigrid](https://github.com/Farama-Foundation/gym-minigrid) environment.
To run EWC agents in the minigrid (MG10) curriculum defined in the paper, use the command below:

```
# Online EWC multi-head policy network (EWC MH)
python train_minigrid_ewc.py ewc --seed 86 --multi_head
```

Note: 
- the full list of commands to run the EWC experiments in the paper can be found in the `paper_experiments.txt` file.

## Note on Continual World
The Continual World benchmark was built on top of the [Meta-World](https://github.com/rlworkgroup/metaworld) benchmark, which comprise of a number of simulated robotics tasks. The originally released Continual World employed the use of version 1 (v1) Meta-World environments. However, the Meta-World v1 environments contained some issues in the reward function (discussed [here](https://github.com/rlworkgroup/metaworld/issues/226) and [here](https://github.com/awarelab/continual_world/issues/2)) which was fixed in the updated v2 environments. Therefore, the experiments in the paper employed the use of the v2 environment for each task in the Continual World.
