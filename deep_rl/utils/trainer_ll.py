import numpy as np
import pickle
import os
import time
import datetime
import torch
from .torch_utils import *

def _itr_log(logger, agent, iteration, dict_logs):
    logger.info('iteration %d, total steps %d, mean/max/min reward %f/%f/%f'%(
        iteration, agent.total_steps,
        np.mean(agent.iteration_rewards),
        np.max(agent.iteration_rewards),
        np.min(agent.iteration_rewards)
    ))
    logger.scalar_summary('last_episode_reward/avg', np.mean(agent.last_episode_rewards))
    logger.scalar_summary('last_episode_reward/std', np.std(agent.last_episode_rewards))
    logger.scalar_summary('last_episode_reward/max', np.max(agent.last_episode_rewards))
    logger.scalar_summary('last_episode_reward/min', np.min(agent.last_episode_rewards))
    logger.scalar_summary('iteration_reward/avg', np.mean(agent.iteration_rewards))
    logger.scalar_summary('iteration_reward/std', np.std(agent.iteration_rewards))
    logger.scalar_summary('iteration_reward/max', np.max(agent.iteration_rewards))
    logger.scalar_summary('iteration_reward/min', np.min(agent.iteration_rewards))

    if hasattr(agent, 'layers_output'):
        for tag, value in agent.layers_output:
            value = value.detach().cpu().numpy()
            value_norm = np.linalg.norm(value, axis=-1)
            logger.scalar_summary('debug/{0}_avg_norm'.format(tag), np.mean(value_norm))
            logger.scalar_summary('debug/{0}_avg'.format(tag), value.mean())
            logger.scalar_summary('debug/{0}_std'.format(tag), value.std())
            logger.scalar_summary('debug/{0}_max'.format(tag), value.max())
            logger.scalar_summary('debug/{0}_min'.format(tag), value.min())

    for key, value in dict_logs.items():
        logger.scalar_summary('debug_extended/{0}_avg'.format(key), np.mean(value))
        logger.scalar_summary('debug_extended/{0}_std'.format(key), np.std(value))
        logger.scalar_summary('debug_extended/{0}_max'.format(key), np.max(value))
        logger.scalar_summary('debug_extended/{0}_min'.format(key), np.min(value))

    return

# metaworld/continualworld
def _itr_log_mw(logger, agent, iteration, dict_logs):
    logger.info('iteration %d, total steps %d, mean/max/min reward %f/%f/%f, ' \
        'mean/max/min success rate %f/%f/%f'%(
        iteration, agent.total_steps,
        np.mean(agent.iteration_rewards),
        np.max(agent.iteration_rewards),
        np.min(agent.iteration_rewards),
        np.mean(agent.iteration_success_rate),
        np.max(agent.iteration_success_rate),
        np.min(agent.iteration_success_rate)
    ))
    logger.scalar_summary('last_episode_reward/avg', np.mean(agent.last_episode_rewards))
    logger.scalar_summary('last_episode_reward/std', np.std(agent.last_episode_rewards))
    logger.scalar_summary('last_episode_reward/max', np.max(agent.last_episode_rewards))
    logger.scalar_summary('last_episode_reward/min', np.min(agent.last_episode_rewards))
    logger.scalar_summary('iteration_reward/avg', np.mean(agent.iteration_rewards))
    logger.scalar_summary('iteration_reward/std', np.std(agent.iteration_rewards))
    logger.scalar_summary('iteration_reward/max', np.max(agent.iteration_rewards))
    logger.scalar_summary('iteration_reward/min', np.min(agent.iteration_rewards))

    logger.scalar_summary('last_episode_success_rate/avg', np.mean(agent.last_episode_success_rate))
    logger.scalar_summary('last_episode_success_rate/std', np.std(agent.last_episode_success_rate))
    logger.scalar_summary('last_episode_success_rate/max', np.max(agent.last_episode_success_rate))
    logger.scalar_summary('last_episode_success_rate/min', np.min(agent.last_episode_success_rate))
    logger.scalar_summary('iteration_success_rate/avg', np.mean(agent.iteration_success_rate))
    logger.scalar_summary('iteration_success_rate/std', np.std(agent.iteration_success_rate))
    logger.scalar_summary('iteration_success_rate/max', np.max(agent.iteration_success_rate))
    logger.scalar_summary('iteration_success_rate/min', np.min(agent.iteration_success_rate))

    if hasattr(agent, 'layers_output'):
        for tag, value in agent.layers_output:
            value = value.detach().cpu().numpy()
            value_norm = np.linalg.norm(value, axis=-1)
            logger.scalar_summary('debug/{0}_avg_norm'.format(tag), np.mean(value_norm))
            logger.scalar_summary('debug/{0}_avg'.format(tag), value.mean())
            logger.scalar_summary('debug/{0}_std'.format(tag), value.std())
            logger.scalar_summary('debug/{0}_max'.format(tag), value.max())
            logger.scalar_summary('debug/{0}_min'.format(tag), value.min())

    for key, value in dict_logs.items():
        logger.scalar_summary('debug_extended/{0}_avg'.format(key), np.mean(value))
        logger.scalar_summary('debug_extended/{0}_std'.format(key), np.std(value))
        logger.scalar_summary('debug_extended/{0}_max'.format(key), np.max(value))
        logger.scalar_summary('debug_extended/{0}_min'.format(key), np.min(value))

    return

# run iterations, lifelong learning
# used by either a baseline agent (with no task knowledge preservation) or
# an agent with knowledge preservation via supermask superposition (ss)
# modules on: PPO agent or PPO agent with supermask
# modules off: detect and resource manager
def run_iterations_w_oracle(agent, tasks_info):
    config = agent.config

    log_path_pm = config.log_dir + '/pm'
    if not os.path.exists(log_path_pm):
        os.makedirs(log_path_pm)
    log_path_tstats = config.log_dir + '/task_stats'
    if not os.path.exists(log_path_tstats):
        os.makedirs(log_path_tstats)
    log_path_eval = config.log_dir + '/eval_stats'
    if not os.path.exists(log_path_eval):
        os.makedirs(log_path_eval)
    random_seed(config.seed)
    agent_name = agent.__class__.__name__

    iteration = 0
    steps = []
    rewards = []
    task_start_idx = 0
    num_tasks = len(tasks_info)
    eval_data_fh = open(config.logger.log_dir + '/eval_metrics.csv', 'a', buffering=1)

    eval_tracker = False
    eval_data = []
    metric_icr = [] # icr => total cumulative reward

    if agent.task.name == config.ENV_METAWORLD or agent.task.name == config.ENV_CONTINUALWORLD:
        itr_log_fn = _itr_log_mw
    else:
        itr_log_fn = _itr_log

    for learn_block_idx in range(config.cl_num_learn_blocks):
        config.logger.info('********** start of learning block {0}'.format(learn_block_idx))
        eval_results = {task_idx:[] for task_idx in range(len(tasks_info))}

        for task_idx, task_info in enumerate(tasks_info):
            config.logger.info('*****start training on task {0}'.format(task_idx))
            config.logger.info('name: {0}'.format(task_info['name']))
            config.logger.info('task: {0}'.format(task_info['task']))
            config.logger.info('task_label: {0}'.format(task_info['task_label']))

            states = agent.task.reset_task(task_info)
            agent.states = config.state_normalizer(states)
            agent.data_buffer.clear()
            agent.task_train_start(task_info['task_label'])
            while True:
                # train step
                dict_logs = agent.iteration()

                iteration += 1
                steps.append(agent.total_steps)
                rewards.append(np.mean(agent.iteration_rewards))

                # logging
                if iteration % config.iteration_log_interval == 0:
                    itr_log_fn(config.logger, agent, iteration, dict_logs)

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

                # evaluation block
                if (agent.config.eval_interval is not None and \
                    iteration % agent.config.eval_interval == 0):
                    config.logger.info('*****agent / evaluation block')
                    _tasks = tasks_info
                    _names = [eval_task_info['name'] for eval_task_info in _tasks]
                    config.logger.info('eval tasks: {0}'.format(', '.join(_names)))
                    eval_data.append(np.zeros(len(_tasks),))
                    for eval_task_idx, eval_task_info in enumerate(_tasks):
                        agent.task_eval_start(eval_task_info['task_label'])
                        eval_states = agent.evaluation_env.reset_task(eval_task_info)
                        agent.evaluation_states = eval_states
                        # performance (perf) can be success rate in (meta-)continualworld or
                        # rewards in other environments
                        perf, eps = agent.evaluate_cl(num_iterations=config.evaluation_episodes)
                        agent.task_eval_end()
                        eval_data[-1][eval_task_idx] = np.mean(perf)
                    _record = np.concatenate([eval_data[-1], np.array(time.time()).reshape(1,)])
                    np.savetxt(eval_data_fh, _record.reshape(1, -1), delimiter=',', fmt='%.4f')
                    del _record
                    icr = eval_data[-1].sum()
                    metric_icr.append(icr)
                    tpot = np.sum(metric_icr)
                    config.logger.info('*****cl evaluation:')
                    config.logger.info('cl eval ICR: {0}'.format(icr))
                    config.logger.info('cl eval TPOT: {0}'.format(tpot))
                    config.logger.scalar_summary('cl_eval/icr', icr)
                    config.logger.scalar_summary('cl_eval/tpot', np.sum(metric_icr))


                # check whether task training has been completed
                task_steps_limit = config.max_steps * (num_tasks * learn_block_idx + task_idx + 1)
                if config.max_steps and agent.total_steps >= task_steps_limit:
                    if agent_name != 'BaselineAgent':
                        config.logger.info('cacheing mask for current task')

                    ret = agent.task_train_end()

                    with open(log_path_tstats + '/%s-%s-online-stats-%s-run-%d-task-%d.bin' % \
                        (agent_name, config.tag, agent.task.name, learn_block_idx+1, task_idx+1), 'wb') as f:
                        pickle.dump({'rewards': rewards[task_start_idx : ], \
                        'steps': steps[task_start_idx : ]}, f)
                    agent.save(log_path_tstats +'/%s-%s-model-%s-run-%d-task-%d.bin' % (agent_name, \
                        config.tag, agent.task.name, learn_block_idx+1, task_idx+1))
                    agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (agent_name, config.tag, \
                        agent.task.name))
                    task_start_idx = len(rewards)

                    with open(log_path_pm + '/%s-%s-precision-matrices-%s-run-%d-task-%d.bin' % \
                        (agent_name, config.tag, agent.task.name, learn_block_idx+1, task_idx+1), 'wb') as f:
                        pickle.dump(ret[0], f)
                    with open(log_path_pm + '/%s-%s-precision-matrices-movavg-%s-run-%d-task-%d.bin' % \
                        (agent_name, config.tag, agent.task.name, learn_block_idx+1, task_idx+1), 'wb') as f:
                        pickle.dump(ret[1], f)
                    break
            # end of while True. current task training
            # evaluate agent across task exposed to agent so far
            config.logger.info('evaluating agent across all tasks exposed so far to agent')
            for j in range(task_idx+1):
                _eval_task = tasks_info[j]
                agent.task_eval_start(_eval_task['task_label'])

                eval_states = agent.evaluation_env.reset_task(tasks_info[j])
                agent.evaluation_states = eval_states
                perf, episodes = agent.evaluate_cl(num_iterations=config.evaluation_episodes)
                eval_results[j] += perf

                agent.task_eval_end()

                with open(log_path_eval+'/rewards-task{0}_{1}.bin'.format(\
                    task_idx+1, j+1), 'wb') as f:
                    pickle.dump(perf, f)
                with open(log_path_eval+'/episodes-task{0}_{1}.bin'.format(\
                    task_idx+1, j+1), 'wb') as f:
                    pickle.dump(episodes, f)
        # end for each task
        print('eval stats')
        with open(log_path_eval + '/eval_full_stats.bin', 'wb') as f: pickle.dump(eval_results, f)

        f = open(log_path_eval + '/eval_stats.csv', 'w')
        f.write('task_id,avg_reward\n')
        for k, v in eval_results.items():
            print('{0}: {1:.4f}'.format(k, np.mean(v)))
            f.write('{0},{1:.4f}\n'.format(k, np.mean(v)))
            config.logger.scalar_summary('zeval/task_{0}/avg_reward'.format(k), np.mean(v))
        f.close()
        config.logger.info('********** end of learning block {0}\n'.format(learn_block_idx))
    # end for learning block
    eval_data_fh.close()
    if len(eval_data) > 0:
        to_save = np.stack(eval_data, axis=0)
        with open(config.logger.log_dir + '/eval_metrics.npy', 'wb') as f:
            np.save(f, to_save)
    agent.close()
    return steps, rewards

