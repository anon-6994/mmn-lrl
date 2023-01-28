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
import matplotlib.pyplot as plt

def _plot_hm_layer_mask_diff(data, title, fname):
    n_tasks = data.shape[0]

    fig = plt.figure(figsize=(9, 9))
    ax = fig.subplots()
    im = ax.imshow(data, cmap='YlGn')
    ax.set_xticks(np.arange(n_tasks), labels=['T{0}'.format(idx) for idx in range(n_tasks)], \
        fontsize=16)
    ax.set_yticks(np.arange(n_tasks), labels=['T{0}'.format(idx) for idx in range(n_tasks)], \
        fontsize=16)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    for i in range(n_tasks):
        for j in range(n_tasks):
         text = ax.text(j, i, '{0:.2f}'.format(data[i, j]), ha='center', \
            va='center', fontsize=16)
    ax.set_title(title, fontsize=20)
    fig.savefig(fname)
    plt.close(fig)


def _plot_hm_betas(data, title, fname):
    n_tasks = data.shape[0]

    fig = plt.figure(figsize=(9, 9))
    ax = fig.subplots()
    im = ax.imshow(data, cmap='YlGn', vmin=0.0, vmax=0.5)
    ax.set_xticks(np.arange(n_tasks), labels=['T{0}'.format(idx) for idx in range(n_tasks)], \
        fontsize=16)
    ax.set_yticks(np.arange(n_tasks), labels=['T{0}'.format(idx) for idx in range(n_tasks)], \
        fontsize=16)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    for i in range(n_tasks):
        for j in range(n_tasks):
         text = ax.text(j, i, '{0:.2f}'.format(data[i, j]), ha='center', \
            va='center', fontsize=16)
    ax.set_title(title, fontsize=20)
    fig.savefig(fname)
    plt.close(fig)

def _plot_hm(data, title, fname):
    n_epi, n_steps = data.shape

    fig = plt.figure(figsize=(9, 9))
    ax = fig.subplots()
    #im = ax.imshow(data, cmap='YlGn')
    im = ax.imshow(data, cmap='YlGn', vmin=0, vmax=2)
    ax.set_xticks(np.arange(n_steps), labels=['S{0}'.format(idx) for idx in range(n_steps)], \
        fontsize=16)
    ax.set_yticks(np.arange(n_epi), labels=['epi_{0}'.format(idx) for idx in range(n_epi)], \
        fontsize=16)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    for i in range(n_epi):
        for j in range(n_steps):
         text = ax.text(j, i, '{0}'.format(data[i, j]), ha='center', \
            va='center', fontsize=16)
    ax.set_title(title, fontsize=20)
    fig.savefig(fname)
    plt.close(fig)

def _plot_hm_policy_output(data, title, fname):
    #n_steps, n_actions = data.shape
    data = data.T
    n_actions, n_steps = data.shape

    fig = plt.figure(figsize=(12, 4))
    ax = fig.subplots()
    #im = ax.imshow(data, cmap='YlGn')
    im = ax.imshow(data, cmap='YlGn', vmin=0, vmax=2)
    ax.set_yticks(np.arange(n_actions), labels=['A{0}'.format(idx) for idx in range(n_actions)], \
        fontsize=16)
    ax.set_xticks(np.arange(n_steps), labels=['S{0}'.format(idx) for idx in range(n_steps)], \
        fontsize=16)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    for i in range(n_actions):
        for j in range(n_steps):
         text = ax.text(j, i, '{0:.2f}'.format(data[i, j]), ha='center', \
            va='center', fontsize=16)
    ax.set_title(title, fontsize=20)
    fig.savefig(fname)
    plt.close(fig)
 
def _eval(agent, tasks_info):
    config = agent.config
    config.logger.info('*****agent / evaluation block')
    _tasks = tasks_info
    _names = [eval_task_info['name'] for eval_task_info in _tasks]
    config.logger.info('eval tasks: {0}'.format(', '.join(_names)))
    eval_data = np.zeros(len(_tasks),)
    tasks_episodes = []
    for eval_task_idx, eval_task_info in enumerate(_tasks):
        agent.task_eval_start(eval_task_info['task_label'])
        eval_states = agent.evaluation_env.reset_task(eval_task_info)
        agent.evaluation_states = eval_states
        # performance (perf) can be success rate in (meta-)continualworld or
        # rewards in other environments
        perf, eps = agent.evaluate_cl(num_iterations=config.evaluation_episodes)
        agent.task_eval_end()
        eval_data[eval_task_idx] = np.mean(perf)
        tasks_episodes.append(eps)
    return eval_data, tasks_episodes
 
def run_episode_te(agent, states, deterministic=True):
    epi_info = {'policy_output': [], 'sampled_action': [], 'log_prob': [], 'entropy': [],
        'value': [], 'agent_action': [], 'reward': [], 'terminal': [], 'state': []}

    with torch.no_grad():
        env = agent.evaluation_env
        state = env.reset()
        if agent.curr_eval_task_label is not None:
            task_label = agent.curr_eval_task_label
        else:
            task_label = env.get_task()['task_label']
            assert False, 'manually set (temporary) breakpoint. code should not get here.'
        total_rewards = 0
        for state in states:
            epi_info['state'].append(state)
            action, output_info = agent.evaluation_action(state, task_label, deterministic)
            for k, v in output_info.items(): epi_info[k].append(v)
    return total_rewards, epi_info

# on a specific task idx, agent does not act in environment based on its actions, rather it acts
# based on states it's given that has been produced by another policy (an optimal policy that
# serves as an auto pilot). we are only interested in investigating how the agent's output for
# the states generated by an optimal policy (auto-pilot)
def _eval_autopilot(agent, tasks_info, autopilot_task_idx, new_task_optimal_path_states):
    config = agent.config
    config.logger.info('*****agent / evaluation block')
    _tasks = tasks_info
    _names = [eval_task_info['name'] for eval_task_info in _tasks]
    config.logger.info('eval tasks: {0}'.format(', '.join(_names)))
    eval_data = np.zeros(len(_tasks),)
    tasks_episodes = []
    for eval_task_idx, eval_task_info in enumerate(_tasks):
        agent.task_eval_start(eval_task_info['task_label'])
        eval_states = agent.evaluation_env.reset_task(eval_task_info)
        agent.evaluation_states = eval_states

        if eval_task_idx != autopilot_task_idx:
            # performance (perf) can be success rate in (meta-)continualworld or
            # rewards in other environments
            perf, eps = agent.evaluate_cl(num_iterations=1) # 1 episode per task
        else:
            perf = []
            eps = []
            _, episode_info = run_episode_te(agent, new_task_optimal_path_states)
            # reward not real and not computed in this analysis, since we are following
            # another policy's optimal path
            perf.append(np.inf)
            eps.append(episode_info)

        agent.task_eval_end()
        eval_data[eval_task_idx] = np.mean(perf)
        tasks_episodes.append(eps)
    return eval_data, tasks_episodes

##### Minigrid environment
'''
ppo, supermask lifelong learning, task boundary (oracle) given
'''
def ppo_ll_minigrid(name, args):
    env_config_path = args.env_config_path

    config = Config()
    config.env_name = name
    config.env_config_path = env_config_path
    config.lr = 0.00015
    config.cl_preservation = 'supermask'
    config.seed = args.seed
    random_seed(config.seed)
    id_ = '-' + args.eval_id if args.eval_id is not None else ''
    exp_id = '-eval-run-{0}-mask-{1}{2}'.format(config.seed, args.new_task_mask, id_)
    del id_
    log_name = name + '-ppo' + '-' + config.cl_preservation + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 4
    # get num_tasks from env_config
    with open(env_config_path, 'r') as f:
        env_config_ = json.load(f)
    num_tasks = len(env_config_['tasks'])
    del env_config_

    #task_fn = lambda log_dir: MiniGridFlatObs(name, env_config_path, log_dir, config.seed, False)
    #config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    eval_task_fn = lambda log_dir: MiniGridFlatObs(name, env_config_path, log_dir, config.seed, True)
    config.eval_task_fn = eval_task_fn
    config.optimizer_fn = lambda params, lr: torch.optim.RMSprop(params, lr=lr)
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_SS(
        state_dim, action_dim, label_dim,
        phi_body=FCBody_SS(state_dim, task_label_dim=label_dim, hidden_units=(200, 200, 200), num_tasks=num_tasks, new_task_mask=args.new_task_mask),
        actor_body=DummyBody_CL(200),
        critic_body=DummyBody_CL(200),
        num_tasks=num_tasks,
        new_task_mask=args.new_task_mask)
    config.policy_fn = SamplePolicy
    #config.state_normalizer = ImageNormalizer()
    # rescale state normaliser: suitable for grid encoding of states in minigrid
    config.state_normalizer = RescaleNormalizer(1./10.)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.1 #0.75
    config.rollout_length = 128
    config.optimization_epochs = 8
    config.num_mini_batches = 64
    config.ppo_ratio_clip = 0.1
    config.iteration_log_interval = 1
    config.gradient_clip = 5
    config.max_steps = args.max_steps
    config.evaluation_episodes = 10
    config.logger = get_logger(log_dir=config.log_dir, file_name='eval-log')
    config.cl_requires_task_label = True

    config.eval_interval = 10
    config.task_ids = np.arange(num_tasks).tolist()

    agent = LLAgent(config)
    config.agent_name = agent.__class__.__name__
    tasks = agent.config.cl_tasks_info
    config.cl_num_learn_blocks = 1
    shutil.copy(env_config_path, config.log_dir + '/env_config.json')

    # useful variables
    agent_name = config.agent_name
    tag = config.tag
    env_name = agent.task.name
    # model with all task knowledge : /path/to/exp/<agent name>-<tag>-model-<env name>.bin
    # model with subset task knowledge: 
    #        path/to/exp/task_stats/<agent name>-<tag>-model-<env name>-run-1-task-<task seen>.bin

    def load_agent(agent, path, num_tasks_learnt):
        agent.load(path)
        # bug fix in loading agent (for linear combination agent). last mask was
        # not consolidated before agent was saved (during training), so do that now.
        for idx in range(num_tasks_learnt):
            agent.seen_tasks[idx] = tasks[idx]['task_label']
        set_num_tasks_learned(agent.network, num_tasks_learnt - 1)
        set_model_task(agent.network, num_tasks_learnt - 1)
        consolidate_mask(agent.network)
        set_num_tasks_learned(agent.network, num_tasks_learnt)
        return agent

    ##### load agent
    model_path = '{0}/{1}-{2}-model-{3}.bin'.format(args.path, agent_name, tag, env_name)
    agent = load_agent(agent, model_path, num_tasks)

    ##### Analysis 1:
    # plot linear combination coefficients if agent employs masking with linear combination.
    # otherwise, if agent is employs masking without linear combination, nothing to plot.
    if args.new_task_mask == 'linear_comb':
        lc_save_path = config.log_dir + '/linear_comb/'
        if not os.path.exists(lc_save_path):
            os.makedirs(lc_save_path)
        if args.algo == 'll_supermask' and args.new_task_mask == 'linear_comb':
            for k, v in agent.network.named_parameters():
                if 'betas' in k:
                    k = k.split('.')
                    if len(k) == 3: # for network.fc_action.betas or network.fc_critic.betas
                        k = '.'.join(k[1:])
                    else: # for network.phi_body.layers.x.betas
                        k = k[2] + k[3] + '.' + k[4]
                    _data = copy.deepcopy(v.detach().cpu())
                    _data[0, 0] = 1.0 # manually set as there is no linear combination for the first task
                    _plot_hm_betas(_data.numpy(), k, '{0}betas_before_softmax_{1}.pdf'.format(lc_save_path, k))
                    with open('{0}betas_before_softmax_{1}.bin'.format(lc_save_path, k), 'wb') as f:
                        pickle.dump(_data.numpy(), f)
                    # apply softmax to get probabilities of co-efficient parameters
                    for _idx in range(_data.shape[0]):
                        _data[_idx, 0:_idx+1] = torch.softmax(_data[_idx, 0:_idx+1], dim=0)
                    _data = _data.numpy()
                    _plot_hm_betas(_data, k, '{0}betas_{1}.pdf'.format(lc_save_path, k))

    ##### Analysis 2:
    # investigate mask correlation across tasks for each layer.
    md_save_path = config.log_dir + '/mask_diff/'
    if not os.path.exists(md_save_path):
        os.makedirs(md_save_path)
    d = {}
    for k, v in agent.network.named_parameters():
        k_split = k.split('.')

        # remove every module that is not a mask (e.g., .weight, .betas)
        try: k_split[-1] = int(k_split[-1])
        except: continue

        if k_split[1] == 'phi_body': new_k = k_split[2] + k_split[3]
        elif k_split[1] == 'fc_action': new_k = k_split[1]
        elif k_split[1] == 'fc_critic': new_k = k_split[1]

        if new_k not in d.keys(): d[new_k] = {}
        d[new_k][k_split[-1]] = copy.deepcopy(v.detach().cpu().numpy())

    with open('{0}mask_parameters.bin'.format(md_save_path), 'wb') as f:
        pickle.dump(d, f)
    config.logger.info(d.keys())
    config.logger.info(d['layers0'].keys())
    #new_d = {}
    for k, v in d.items():
        diff_norm_data = np.zeros((num_tasks, num_tasks))
        diff_mean_data = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(num_tasks):
                diff_norm_data[i, j] = np.linalg.norm(d[k][i] - d[k][j])
                #diff_mean_data[i, j] = np.mean(np.abs(d[k][i] - d[k][j])) * 100.
                diff_mean_data[i, j] = np.mean(np.abs(d[k][i] - d[k][j]))
        #new_d[k] = diff_norm_data
        _plot_hm_layer_mask_diff(diff_norm_data, \
            'Mask correlation for across tasks for {0}'.format(k), \
            '{0}layer_{1}_mask_diff_norm.pdf'.format(md_save_path, k))
        _plot_hm_layer_mask_diff(diff_mean_data, \
            'Mask correlation for across tasks for {0}'.format(k), \
            '{0}layer_{1}_mask_diff_mean.pdf'.format(md_save_path, k))

    ##### Analysis 3:
    # evaluate agent on all tasks in the curriculum
    ge_save_path = config.log_dir + '/eval/'
    if not os.path.exists(ge_save_path):
        os.makedirs(ge_save_path)
    eval_data, ret = _eval(agent, tasks)
    with open(ge_save_path + 'eval_summary.bin', 'wb') as f: pickle.dump(eval_data, f)
    with open(ge_save_path + 'eval_full_stats.bin', 'wb') as f: pickle.dump(ret, f)
    config.logger.info('General evaluation:')
    config.logger.info(eval_data)
    config.logger.info('\n')

    # target exploration (block of code below) not required as agent as seen all
    # tasks in the curriculum, therefore nothing to test on.

    # close agent.
    agent.close()
    del agent

    if args.te_num_tasks_seen is None:
        # end analysis here. no need for targeted exploration analysis
        return
    else:
        # store optimal trajectory of states for autopilot evaluation if set in targeted
        # exploration analysis
        ret_optimal = ret

    ##### Analysis 4: Targeted exploration
    # targeted exploration for new task (the next task, after seen tasks, in the curriculum)
    # (i.e., agent's behaviour on the new task before any training is performed).
    te_num_tasks_seen = args.te_num_tasks_seen
    agent = LLAgent(config)
    model_path = '{0}/task_stats/{1}-{2}-model-{3}-run-1-task-{4}.bin'.format(\
        args.path, agent_name, tag, env_name, te_num_tasks_seen)
    agent = load_agent(agent, model_path, te_num_tasks_seen)

    te_save_path = config.log_dir + '/targeted_exploration/'
    if not os.path.exists(te_save_path):
        os.makedirs(te_save_path)
    new_task_idx = te_num_tasks_seen #0 index notation means we don't need to add 1 to get next task.


    # slight detour: policy path for optimal policy in the new task
    title = 'Task {0} (new task); {1}'.format(new_task_idx, 'Optimal Policy')
    fname = 'task_{0}_new_optimal'.format(new_task_idx)
    fname = te_save_path + fname
    policy_output = ret_optimal[new_task_idx][0]['policy_output']
    policy_output = [x.view(-1) for x in policy_output]
    policy_output = torch.stack(policy_output, dim=0)
    policy_output = torch.softmax(policy_output, dim=1)
    _plot_hm_policy_output(policy_output, title, fname + '_policy_output.pdf')


    # set initial linear combination co-efficients for the new task
    agent.task_train_start(tasks[new_task_idx]['task_label'])

    # only evaluate on subset tasks: up to the new task
    tasks_subset = tasks[ : new_task_idx + 1]

    # log
    config.logger.info('targeted evaluation:')

    if args.te_autopilot:
        states = ret_optimal[new_task_idx][0]['state'] # states in epiosde 0
        eval_data, ret = _eval_autopilot(agent, tasks_subset, new_task_idx, states) # 1 epi per task
    else:
        eval_data, ret = _eval(agent, tasks_subset) # 10 episodes per task
        config.logger.info(eval_data)
        config.logger.info('\n')

    with open(te_save_path + 'eval_summary.bin', 'wb') as f: pickle.dump(eval_data, f)
    with open(te_save_path + 'eval_full_stats.bin', 'wb') as f: pickle.dump(ret, f)
    for task_idx, task_data in enumerate(ret):
        action_buffer_sampled = []
        action_buffer_final = []

        if task_idx < new_task_idx:
            title = 'Task {0}; previously learnt mask'.format(task_idx)
            fname = 'task_{0}_already_learnt'.format(task_idx)
        elif task_idx == new_task_idx:
            mask_type = 'MASK RI'  if args.new_task_mask == 'random' else 'MASK LC'
            title = 'Task {0} (new task); {1}'.format(task_idx, mask_type)
            fname = 'task_{0}_new'.format(task_idx)
        else:
            continue

        fname = te_save_path + fname

        for episode_idx, episode_data in enumerate(task_data):
            # episode_data: dictionary with keys such as rewards, log_prob, agent_action 
            # and so on. Each key has a value that is a list with length of an episode
            config.logger.info('task {0}, epsiode {1}, rewards: {2}'.format(task_idx, \
                episode_idx, np.sum(episode_data['reward'])))
            action_buffer_sampled.append(episode_data['sampled_action'])
            action_buffer_final.append(episode_data['agent_action'])
        action_buffer_sampled = [[int(x.detach().cpu()) for x in ep] for ep in action_buffer_sampled]
        action_buffer_sampled = np.asarray(action_buffer_sampled, dtype=np.uint8)
        action_buffer_final = np.asarray(action_buffer_final, dtype=np.uint8)
        action_buffer_sampled[ : , 0] = 0.
        action_buffer_final[ : , 0] = 0.
        action_buffer_sampled[ : , -1] = 0.
        action_buffer_final[ : , -1] = 0.
        _plot_hm(action_buffer_sampled, title, fname + '_sampled.pdf')
        _plot_hm(action_buffer_final, title, fname + '_deterministic.pdf')

        # take episode 0 and plot the policy output against steps as heatmap
        policy_output = task_data[0]['policy_output']
        policy_output = [x.view(-1) for x in policy_output]
        policy_output = torch.stack(policy_output, dim=0)
        policy_output = torch.softmax(policy_output, dim=1)
        _plot_hm_policy_output(policy_output, title, fname + '_policy_output.pdf')
        # take episode 0 and plot the policy output entropy aginst steps
        policy_entropy = task_data[0]['entropy']
        policy_entropy = [x.view(-1) for x in policy_entropy]
        policy_entropy = torch.cat(policy_entropy, dim=0)
        fig, ax = plt.subplots()
        ax.plot(policy_entropy)
        ax.set_title(title, fontsize=18)
        ax.set_xlabel('Steps', fontsize=18)
        ax.set_ylabel('Entropy', fontsize=18)
        fig.set_tight_layout(True)
        fig.savefig(fname + '_entropy.pdf')
        plt.close(fig)

if __name__ == '__main__':
    mkdir('log')
    set_one_thread()
    #select_device(0) # -1 is CPU, a positive integer is the index of GPU
    select_device(-1) # -1 is CPU, a positive integer is the index of GPU

    parser = argparse.ArgumentParser()
    parser.add_argument('algo', help='algorithm to run')
    parser.add_argument('path')
    parser.add_argument('--env_name', help='name of the evaluation environment. ' \
        'minigrid and ctgraph currently supported', default='minigrid')
    parser.add_argument('--env_config_path', help='path to environment config', \
        default='./env_configs/minigrid_10.json')
    parser.add_argument('--max_steps', help='maximum number of training steps per task.', \
        default=51200*20, type=int)
    parser.add_argument('--new_task_mask', help='', \
        default='random', type=str)
    parser.add_argument('--seed', help='seed for experiment', default=8379, type=int)
    parser.add_argument('--te_num_tasks_seen', help='number of tasks that the agent has been ' \
        'trained on in the curriculum (for targeted exploration analysis', default=None, type=int)
    parser.add_argument('--te_autopilot', help='flag to determine whether targeted exploration' \
        'analysis is based on the agent\'s own behaviour and output in the task or the '\
        'investigation of the agent\'s output following a set of states that based on a '\
        'trajectory defined by another policy (an optimal policy or an oracle', default=False, \
        action='store_true')
    parser.add_argument('--eval_id', help='id of evaluation run', default=None, type=str)
    args = parser.parse_args()

    if args.env_config_path is None:
        paths_ = args.path.split('/')
        if paths_[-2] == 'task_stats': base_path = '/'.join(paths_[ : -2])
        else: base_path = '/'.join(paths_[ : -1])
        args.env_config_path = base_path + '/env_config.json'
        del base_path

    if args.env_name == 'minigrid':
        name = Config.ENV_MINIGRID
        if args.algo == 'baseline':
            ppo_baseline_minigrid(name, args)
        elif args.algo == 'll_supermask':
            ppo_ll_minigrid(name, args)
        else:
            raise ValueError('algo {0} not implemented'.format(args.algo))
    else:
        raise ValueError('--env_name {0} not implemented'.format(args.env_name))
