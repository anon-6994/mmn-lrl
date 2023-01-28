#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle
import os
import datetime
import torch
from .torch_utils import *
#from io import BytesIO
#import scipy.misc
#import torchvision

try:
    # python >= 3.5
    from pathlib import Path
except:
    # python == 2.7
    from pathlib2 import Path

def run_episodes(agent): # run episodes in single task setting
    config = agent.config
    random_seed(config.seed)
    window_size = 100
    ep = 0
    rewards = []
    steps = []
    avg_test_rewards = []
    agent_type = agent.__class__.__name__
    task_name = agent.task.name

    while True:
        ep += 1
        reward, step = agent.episode()
        rewards.append(reward)
        steps.append(step)
        avg_reward = np.mean(rewards[-window_size:])
        config.logger.info('episode %d, reward %f, avg reward %f, total steps %d, episode step %d' %
            (ep, reward, avg_reward, agent.total_steps, step))
        # tensorboard log
        config.logger.scalar_summary('reward', reward)
        config.logger.scalar_summary('max reward', np.max(rewards[-window_size:]))
        config.logger.scalar_summary('avg reward', avg_reward)
        # other logs
        if config.save_interval and ep % config.save_interval == 0:
            with open(config.log_dir + '/%s-%s-online-stats-%s.bin' % (
                    agent_type, config.tag, task_name), 'wb') as f:
                pickle.dump([steps, rewards], f)
            agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (agent_type, config.tag, task_name))
            np.save(config.log_dir + '/rewards', rewards)

        if config.episode_limit and ep > config.episode_limit:
            with open(config.log_dir + '/%s-%s-online-stats-%s.bin' % (
                    agent_type, config.tag, task_name), 'wb') as f:
                pickle.dump([steps, rewards], f)
            agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (agent_type, config.tag, task_name))
            break

        if config.max_steps and agent.total_steps > config.max_steps:
            with open(config.log_dir + '/%s-%s-online-stats-%s.bin' % (
                    agent_type, config.tag, task_name), 'wb') as f:
                pickle.dump([steps, rewards], f)
            agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (agent_type, config.tag, task_name))
            break

    agent.close()
    return steps, rewards, avg_test_rewards

def run_episodes_cl(agent, tasks_info): # run episodes in continual learning (mulitple tasks) setting
    config = agent.config
    random_seed(config.seed)
    window_size = 100
    ep = 0
    rewards = []
    steps = []
    avg_test_rewards = []
    agent_name = agent.__class__.__name__
    task_name = agent.task.name

    task_start_idx = 0
    eval_results = {task_idx:[] for task_idx in range(len(tasks_info))}
    for task_idx, task_info in enumerate(tasks_info):
        config.logger.info('\nstart training on task {0}'.format(task_idx))
        states = agent.task.reset_task(task_info)
        agent.states = states
        del agent.replay
        agent.replay = agent.config.replay_fn()
        del agent.policy
        agent.policy = agent.config.policy_fn()

        while True:
            ep += 1
            reward, step = agent.episode()
            rewards.append(reward)
            steps.append(step)
            avg_reward = np.mean(rewards[-window_size:])
            if ep % config.episode_log_interval == 0:
                config.logger.info('episode %d, reward %f, avg reward %f, total steps %d,' \
                    'episode step %d' % (ep, reward, avg_reward, agent.total_steps, step))
                # tensorboard log
                config.logger.scalar_summary('reward', reward)
                config.logger.scalar_summary('max reward', np.max(rewards[-window_size:]))
                config.logger.scalar_summary('avg reward', avg_reward)
            # other logs
            if config.save_interval and ep % config.save_interval == 0:
                with open(config.log_dir + '/%s-%s-online-stats-%s.bin' % (
                        agent_name, config.tag, task_name), 'wb') as f:
                    pickle.dump([steps, rewards], f)
                agent.save(config.log_dir+'/%s-%s-model-%s.bin'%(agent_name, config.tag, task_name))
                np.save(config.log_dir + '/rewards', rewards)

            if config.episode_limit and ep > (config.episode_limit*(task_idx+1)):
                with open(config.log_dir + '/%s-%s-online-stats-%s.bin' % (
                        agent_name, config.tag, task_name), 'wb') as f:
                    pickle.dump([steps, rewards], f)
                agent.save(config.log_dir+'/%s-%s-model-%s.bin'%(agent_name, config.tag, task_name))
                break

            if config.max_steps and agent.total_steps > (config.max_steps*(task_idx+1)):
                with open(config.log_dir + '/%s-%s-online-stats-%s.bin' % (
                        agent_name, config.tag, task_name), 'wb') as f:
                    pickle.dump([steps, rewards], f)
                agent.save(config.log_dir+'/%s-%s-model-%s.bin'%(agent_name, config.tag, task_name))
                break
        config.logger.info('preserving learned weights for current task')
        config.logger.info('epsilon greedy status: {0}'.format(agent.policy.epsilon.current))
        ret = agent.consolidate()
        with open(config.log_dir + '/%s-%s-precision-matrices-%s-task-%d.bin' % \
            (agent_name, config.tag, task_name, task_idx+1), 'wb') as f:
            pickle.dump(ret[0], f)
        with open(config.log_dir + '/%s-%s-precision-matrices-movavg-%s-task-%d.bin' % \
            (agent_name, config.tag, task_name, task_idx+1), 'wb') as f:
            pickle.dump(ret[1], f)
        # evaluate agent across task exposed to agent so far
        config.logger.info('evaluating agent across all tasks exposed so far to agent')
        for j in range(task_idx+1):
            eval_states = agent.evaluation_env.reset_task(tasks_info[j])
            agent.evaluation_states = eval_states
            rewards, episodes = agent.evaluate_cl(num_iterations=config.evaluation_episodes)
            eval_results[j] += rewards
            with open(config.log_dir+'/rewards-task{0}_{1}.bin'.format(task_idx+1, j+1), 'wb') as f:
                pickle.dump(rewards, f)
            with open(config.log_dir+'/episodes-task{0}_{1}.bin'.format(task_idx+1, j+1), 'wb') as f:
                pickle.dump(episodes, f)

    agent.close()
    for k, v in eval_results.items():
        print('{0}: {1}'.format(k, np.mean(v)))
    print(eval_results)
    return steps, rewards, avg_test_rewards

def run_iterations(agent): # run iterations single task setting
    config = agent.config
    random_seed(config.seed)
    agent_name = agent.__class__.__name__
    iteration = 0
    steps = []
    rewards = []

    while True:
        agent.iteration()
        steps.append(agent.total_steps)
        rewards.append(np.mean(agent.last_episode_rewards))
        if iteration % config.iteration_log_interval == 0:
            config.logger.info('total steps %d, mean/max/min reward %f/%f/%f' % (
                agent.total_steps, np.mean(agent.last_episode_rewards),
                np.max(agent.last_episode_rewards),
                np.min(agent.last_episode_rewards)
            ))
            config.logger.scalar_summary('avg reward', np.mean(agent.last_episode_rewards))
            config.logger.scalar_summary('max reward', np.max(agent.last_episode_rewards))
            config.logger.scalar_summary('min reward', np.min(agent.last_episode_rewards))

        #if iteration % (config.iteration_log_interval * 100) == 0:
        if iteration % (config.iteration_log_interval) == 0:
            with open(config.log_dir + '/%s-%s-online-stats-%s.bin' % \
                (agent_name, config.tag, agent.task.name), 'wb') as f:
                pickle.dump({'rewards': rewards, 'steps': steps}, f)
            agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (agent_name, config.tag, \
                agent.task.name))
            for tag, value in agent.network.named_parameters():
                tag = tag.replace('.', '/')
                config.logger.histo_summary(tag, value.data.cpu().numpy())
        iteration += 1
        if config.max_steps and agent.total_steps >= config.max_steps:
            with open(config.log_dir + '/%s-%s-online-stats-%s.bin' % \
                (agent_name, config.tag, agent.task.name), 'wb') as f:
                pickle.dump({'rewards': rewards, 'steps': steps}, f)
            agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (agent_name, config.tag, \
                agent.task.name))
            agent.close()
            break
    agent.close()
    return steps, rewards

def run_iterations_cl(agent, tasks_info): #run iterations continual learning (mulitple tasks) setting
    config = agent.config

    log_path_pm = config.log_dir + '/pm'
    if not os.path.exists(log_path_pm):
        os.makedirs(log_path_pm)
    log_path_tstats = config.log_dir + '/task_stats'
    if not os.path.exists(log_path_tstats):
        os.makedirs(log_path_tstats)
    log_path_eval = config.log_dir + '/eval'
    if not os.path.exists(log_path_eval):
        os.makedirs(log_path_eval)
    # save neuromodulated (hyper) nets before training
    try:
        agent.nm_nets # check that nm_nets is an attribute in agent
        with open(config.log_dir + '/nm_nets_before_train.bin', 'wb') as f:
            pickle.dump(agent.nm_nets, f)
    except:
        pass

    random_seed(config.seed)
    agent_name = agent.__class__.__name__

    iteration = 0
    steps = []
    rewards = []
    task_start_idx = 0
    num_tasks = len(tasks_info)

    for learn_block_idx in range(config.cl_num_learn_blocks):
        config.logger.info('********** start of learning block {0}'.format(learn_block_idx))
        eval_results = {task_idx:[] for task_idx in range(len(tasks_info))}
        for task_idx, task_info in enumerate(tasks_info):

            config.logger.info('*****start training on task {0}'.format(task_idx))
            config.logger.info('task: {0}'.format(task_info['task']))
            config.logger.info('task_label: {0}'.format(task_info['task_label']))

            states = agent.task.reset_task(task_info)
            agent.states = config.state_normalizer(states)
            agent.data_buffer.clear()
            agent.task_train_start()
            while True:
                avg_grad_norm = agent.iteration()
                steps.append(agent.total_steps)
                rewards.append(np.mean(agent.last_episode_rewards))
                if iteration % config.iteration_log_interval == 0:
                    config.logger.info('iteration %d, total steps %d, mean/max/min reward %f/%f/%f'%(
                        iteration, agent.total_steps, np.mean(agent.last_episode_rewards),
                        np.max(agent.last_episode_rewards),
                        np.min(agent.last_episode_rewards)
                    ))
                    config.logger.scalar_summary('avg reward', np.mean(agent.last_episode_rewards))
                    config.logger.scalar_summary('max reward', np.max(agent.last_episode_rewards))
                    config.logger.scalar_summary('min reward', np.min(agent.last_episode_rewards))
                    config.logger.scalar_summary('avg grad norm', avg_grad_norm)

                if iteration % (config.iteration_log_interval) == 0:
                    with open(config.log_dir + '/%s-%s-online-stats-%s.bin' % \
                        (agent_name, config.tag, agent.task.name), 'wb') as f:
                        pickle.dump({'rewards': rewards, 'steps': steps}, f)
                    agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (agent_name, config.tag, \
                        agent.task.name))
                    for tag, value in agent.network.named_parameters():
                        tag = tag.replace('.', '/')
                        config.logger.histo_summary(tag, value.data.cpu().numpy())
                    if hasattr(agent, 'layers_output'):
                        for tag, value in agent.layers_output:
                            tag = 'layer_output/' + tag
                            config.logger.histo_summary(tag, value.data.cpu().numpy())

                iteration += 1
                task_steps_limit = config.max_steps * (num_tasks * learn_block_idx + task_idx + 1)
                if config.max_steps and agent.total_steps >= task_steps_limit:
                    with open(log_path_tstats + '/%s-%s-online-stats-%s-run-%d-task-%d.bin' % \
                        (agent_name, config.tag, agent.task.name, learn_block_idx+1, task_idx+1), 'wb') as f:
                        pickle.dump({'rewards': rewards[task_start_idx : ], \
                        'steps': steps[task_start_idx : ]}, f)
                    agent.save(log_path_tstats +'/%s-%s-model-%s-run-%d-task-%d.bin' % (agent_name, \
                        config.tag, agent.task.name, learn_block_idx+1, task_idx+1))
                    agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (agent_name, config.tag, \
                        agent.task.name))
                    task_start_idx = len(rewards)
                    break
            config.logger.info('preserving learned weights for current task')
            ret = agent.task_train_end() # consolidate is implicitly called in this method
            tasks_info[task_idx]['task_label_agent'] = ret['task_label_agent']
            ret = ret['consolidate']
            with open(log_path_pm + '/%s-%s-precision-matrices-%s-run-%d-task-%d.bin' % \
                (agent_name, config.tag, agent.task.name, learn_block_idx+1, task_idx+1), 'wb') as f:
                pickle.dump(ret[0], f)
            with open(log_path_pm + '/%s-%s-precision-matrices-movavg-%s-run-%d-task-%d.bin' % \
                (agent_name, config.tag, agent.task.name, learn_block_idx+1, task_idx+1), 'wb') as f:
                pickle.dump(ret[1], f)
            # evaluate agent across task exposed to agent so far
            config.logger.info('evaluating agent across all tasks exposed so far to agent')
            for j in range(task_idx+1):
                eval_states = agent.evaluation_env.reset_task(tasks_info[j])
                agent.evaluation_states = eval_states
                rewards, episodes = agent.evaluate_cl(num_iterations=config.evaluation_episodes)
                eval_results[j] += rewards

                with open(config.log_dir+'/rewards-task{0}_{1}.bin'.format(\
                    task_idx+1, j+1), 'wb') as f:
                    pickle.dump(rewards, f)
                with open(config.log_dir+'/episodes-task{0}_{1}.bin'.format(\
                    task_idx+1, j+1), 'wb') as f:
                    pickle.dump(episodes, f)
        print('eval stats')
        with open(config.log_dir + '/eval_full_stats.bin', 'wb') as f: pickle.dump(eval_results, f)

        f = open(config.log_dir + '/eval_stats.csv', 'w')
        f.write('task_id,avg_reward\n')
        for k, v in eval_results.items():
            print('{0}: {1:.4f}'.format(k, np.mean(v)))
            f.write('{0},{1:.4f}\n'.format(k, np.mean(v)))
            config.logger.scalar_summary('zeval/task_{0}/avg_reward'.format(k), np.mean(v))
        f.close()
        config.logger.info('********** end of learning block {0}\n'.format(learn_block_idx))

    # save neuromodulated (hyper) nets after training
    try:
        agent.nm_nets # check that nm_nets is an attribute in agent
        with open(config.log_dir + '/nm_nets_after_train.bin', 'wb') as f:
            pickle.dump(agent.nm_nets, f)
    except:
        pass

    agent.close()
    return steps, rewards

def run_evals_cl(agent, tasks_info, num_evals): 
    #run evaluations of agent across multiple task it has been trained (exposed to)
    # in continual learning, weight preservation setting
    config = agent.config
    random_seed(config.seed)
    rewards = []
    episodes = []
    for task_idx in range(len(tasks_info)):
        eval_states = agent.evaluation_env.reset_task(tasks_info[task_idx])
        agent.evaluation_states = eval_states
        task_rewards, task_episodes = agent.evaluate_cl(num_iterations=config.evaluation_episodes)
        rewards.append(task_rewards)
        episodes.append(task_episodes)
        config.logger.info('task {0} / mean reward(across episodes): {1}'.format(
            task_idx+1, np.mean(task_rewards)))
    agent.close()
    with open(config.log_dir + '/rewards.bin', 'wb') as f:
        pickle.dump(rewards, f)
    with open(config.log_dir + '/episodes.bin', 'wb') as f:
        pickle.dump(episodes, f)
    for task_idx, task_rewards in enumerate(rewards):
        print('task {0}'.format(task_idx+1))
        print(task_rewards)
    return rewards

def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")

def get_default_log_dir(name):
    return './log/%s/%s' % (name, get_time_str())

def sync_grad(target_network, src_network):
    for param, src_param in zip(target_network.parameters(), src_network.parameters()):
        param._grad = src_param.grad.clone()

def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

class Batcher:
    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.num_entries = len(data[0])
        self.reset()

    def reset(self):
        self.batch_start = 0
        self.batch_end = self.batch_start + self.batch_size

    def end(self):
        return self.batch_start >= self.num_entries

    def next_batch(self):
        batch = []
        for d in self.data:
            batch.append(d[self.batch_start: self.batch_end])
        self.batch_start = self.batch_end
        self.batch_end = min(self.batch_start + self.batch_size, self.num_entries)
        return batch

    def shuffle(self):
        indices = np.arange(self.num_entries)
        np.random.shuffle(indices)
        self.data = [d[indices] for d in self.data]
