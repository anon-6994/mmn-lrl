#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

'''
lifelong (continual) learning experiments using supermask
superpostion algorithm in RL.
https://arxiv.org/abs/2006.14769
'''

import json
import copy
import shutil
import matplotlib
matplotlib.use("Pdf")
from deep_rl import *
import os
import argparse

##### ContinualWorld environment
'''
ppo, baseline (no lifelong learning), task boundary (oracle) given
'''
def ppo_baseline_continualworld(name, args):
    env_config_path = args.env_config_path

    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 5e-4
    config.cl_preservation = 'baseline'
    config.seed = args.seed
    random_seed(config.seed)
    exp_id = '-{0}'.format(config.seed)
    log_name = name + '-ppo' + '-' + config.cl_preservation + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 1 # 4 (if 4, rollout should be 5120 * 2.5)

    # get num_tasks from env_config
    with open(env_config_path, 'r') as f:
        env_config_ = json.load(f)
    num_tasks = len(env_config_['tasks'])
    del env_config_

    task_fn = lambda log_dir: ContinualWorld(name, env_config_path, log_dir, config.seed)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    eval_task_fn = lambda log_dir: ContinualWorld(name, env_config_path, log_dir, config.seed)
    config.eval_task_fn = eval_task_fn
    config.optimizer_fn = lambda params, lr: torch.optim.Adam(params, lr=lr)
    config.network_fn = lambda state_dim, action_dim, label_dim: GaussianActorCriticNet_CL(
        state_dim, action_dim, label_dim,
        phi_body=DummyBody_CL(state_dim, task_label_dim=label_dim),
        actor_body=FCBody_CL(state_dim + label_dim, hidden_units=(128, 128), gate=torch.tanh),
        critic_body=FCBody_CL(state_dim + label_dim, hidden_units=(128, 128), gate=torch.tanh))
    config.policy_fn = SamplePolicy
    #config.state_normalizer = RescaleNormalizer(1.) # no rescaling
    config.state_normalizer = RunningStatsNormalizer()
    config.reward_normalizer = RewardRunningStatsNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.97
    config.entropy_weight = 5e-3
    config.rollout_length = 512 * 10 # (i.e., 512 * 2.5, if num_workers is set to 4) 
    config.optimization_epochs = 16
    config.num_mini_batches = 160 # with rollout of 5120, 160 mini_batch gives 32 samples per batch
    config.ppo_ratio_clip = 0.2
    config.iteration_log_interval = 1
    config.gradient_clip = 5
    config.max_steps = args.max_steps
    config.evaluation_episodes = 10
    config.logger = get_logger(log_dir=config.log_dir, file_name='train-log')
    config.cl_requires_task_label = True

    config.eval_interval = 200
    config.task_ids = np.arange(num_tasks).tolist()

    agent = BaselineAgent(config)
    config.agent_name = agent.__class__.__name__
    tasks = agent.config.cl_tasks_info
    config.cl_num_learn_blocks = 1
    shutil.copy(env_config_path, config.log_dir + '/env_config.json')
    with open('{0}/tasks_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)
    run_iterations_w_oracle(agent, tasks)
    with open('{0}/tasks_info_after_train.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)
    # save config
    with open('{0}/config.json'.format(config.log_dir), 'w') as f:
        dict_config = vars(config)
        for k in dict_config.keys():
            if not isinstance(dict_config[k], int) \
            and not isinstance(dict_config[k], float) and dict_config[k] is not None:
                dict_config[k] = str(dict_config[k])
        json.dump(dict_config, f)
'''
ppo, supermask lifelong learning, task boundary (oracle) given
'''
def ppo_ll_continualworld(name, args):
    env_config_path = args.env_config_path

    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 5e-4
    config.cl_preservation = 'supermask'
    config.seed = args.seed
    random_seed(config.seed)
    exp_id = '-{0}-mask-{1}'.format(config.seed, args.new_task_mask)
    log_name = name + '-ppo' + '-' + config.cl_preservation + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 1 # 4 (if 4, rollout should be 5120 * 2.5)

    # get num_tasks from env_config
    with open(env_config_path, 'r') as f:
        env_config_ = json.load(f)
    num_tasks = len(env_config_['tasks'])
    del env_config_

    task_fn = lambda log_dir: ContinualWorld(name, env_config_path, log_dir, config.seed)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    eval_task_fn = lambda log_dir: ContinualWorld(name, env_config_path, log_dir, config.seed)
    config.eval_task_fn = eval_task_fn
    config.optimizer_fn = lambda params, lr: torch.optim.Adam(params, lr=lr)
    config.network_fn = lambda state_dim, action_dim, label_dim: GaussianActorCriticNet_SS(
        state_dim, action_dim, label_dim,
        phi_body=DummyBody_CL(state_dim, task_label_dim=label_dim),
        actor_body=FCBody_SS(state_dim + label_dim, hidden_units=(128, 128), gate=torch.tanh, \
            discrete_mask=False, num_tasks=num_tasks, new_task_mask=args.new_task_mask),
        critic_body=FCBody_SS(state_dim + label_dim, hidden_units=(128, 128), gate=torch.tanh, \
            discrete_mask=False, num_tasks=num_tasks, new_task_mask=args.new_task_mask),
        num_tasks=num_tasks, new_task_mask=args.new_task_mask)
    config.policy_fn = SamplePolicy
    #config.state_normalizer = RescaleNormalizer(1.) # no rescaling
    config.state_normalizer = RunningStatsNormalizer()
    config.reward_normalizer = RewardRunningStatsNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.97
    config.entropy_weight = 5e-3
    config.rollout_length = 512 * 10 # (i.e., 512 * 2.5, if num_workers is set to 4)
    config.optimization_epochs = 16
    config.num_mini_batches = 160 # with rollout of 5120, 160 mini_batch gives 32 samples per batch
    config.ppo_ratio_clip = 0.2
    config.iteration_log_interval = 1
    config.gradient_clip = 5
    config.max_steps = args.max_steps
    config.evaluation_episodes = 10
    config.logger = get_logger(log_dir=config.log_dir, file_name='train-log')
    config.cl_requires_task_label = True

    config.eval_interval = 200
    config.task_ids = np.arange(num_tasks).tolist()

    agent = LLAgent(config)
    config.agent_name = agent.__class__.__name__
    tasks = agent.config.cl_tasks_info
    config.cl_num_learn_blocks = 1
    shutil.copy(env_config_path, config.log_dir + '/env_config.json')
    with open('{0}/tasks_info.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)
    run_iterations_w_oracle(agent, tasks)
    with open('{0}/tasks_info_after_train.bin'.format(config.log_dir), 'wb') as f:
        pickle.dump(tasks, f)
    # save config
    with open('{0}/config.json'.format(config.log_dir), 'w') as f:
        dict_config = vars(config)
        for k in dict_config.keys():
            if not isinstance(dict_config[k], int) \
            and not isinstance(dict_config[k], float) and dict_config[k] is not None:
                dict_config[k] = str(dict_config[k])
        json.dump(dict_config, f)

if __name__ == '__main__':
    mkdir('log')
    set_one_thread()
    select_device(0) # -1 is CPU, a positive integer is the index of GPU

    parser = argparse.ArgumentParser()
    parser.add_argument('algo', help='algorithm to run')
    parser.add_argument('--env_name', help='name of the evaluation environment. ' \
        'minigrid and ctgraph currently supported', default='continualworld')
    parser.add_argument('--env_config_path', help='path to environment config', \
        default='./env_configs/continualworld_10.json')
    parser.add_argument('--max_steps', help='maximum number of training steps per task.', \
        default=10_240_000, type=int)
    parser.add_argument('--new_task_mask', help='', \
        default='random', type=str)
    parser.add_argument('--seed', help='seed for the experiment', default=8379, type=int)
    args = parser.parse_args()

    if args.env_name == 'continualworld':
        name = Config.ENV_CONTINUALWORLD
        if args.algo == 'baseline':
            ppo_baseline_continualworld(name, args)
        elif args.algo == 'll_supermask':
            ppo_ll_continualworld(name, args)
        else:
            raise ValueError('algo {0} not implemented'.format(args.algo))
    else:
        raise ValueError('--env_name {0} not implemented'.format(args.env_name))
