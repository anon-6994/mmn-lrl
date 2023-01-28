#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import copy
from .atari_wrapper import *
import multiprocessing as mp
import sys
from .bench import Monitor
from ..utils import *
import uuid
import json
import itertools

# fix to enable running the code on MacOS using python>=3.8
# spawn multiprocessing start method fails to run the lambda
# env initialisation function.
if mp.get_start_method() == 'spawn':
    #print('setting multiprocessing start method to use fork')
    mp.set_start_method('fork', force=True)

class BaseTask:
    def __init__(self):
        pass

    def set_monitor(self, env, log_dir):
        if log_dir is None:
            return env
        mkdir(log_dir)
        return Monitor(env, '%s/%s' % (log_dir, uuid.uuid4()))

    def reset(self):
        #print 'base task reset called'
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        #print 'base task step called'
        #print self.env
        #print done
        if done:
            next_state = self.env.reset()
        return next_state, reward, done, info

    def seed(self, random_seed):
        return self.env.seed(random_seed)

class ClassicalControl(BaseTask):
    def __init__(self, name, max_steps=200, log_dir=None):
        # Removed from name = "Cart_Pole-v0"
        BaseTask.__init__(self)
        self.name = name
        self.env = gym.make(self.name)
        self.env._max_episode_steps = max_steps
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

class DynamicGrid(BaseTask):
    def __init__(self, name, env_config_path=None, log_dir=None, seed=None, max_steps=100):
        BaseTask.__init__(self)
        self.name = name
        import dynamic_grid
        self.env = gym.make(self.name)
        self.env._max_episode_steps = max_steps
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape
        self.env = self.set_monitor(self.env, log_dir)

        if seed is not None:
            self.seed(seed)
        np.random.seed(seed)
        task_change_points = 3 # reward fn, transition fn, and state space

        # method 1: manually specify tasks based on change points
        from itertools import product
        num_tasks = 2 ** task_change_points
        change_matrix = np.array([[0, 0, 0], # base task
                                [1, 0, 0], # change goal location (reward function) only
                                [0, 1, 0], # change transition function only
                                [0, 0, 1], # change input distribution only
                                [1, 1, 0], # change reward fn and transition fn
                                [1, 0, 1], # change reward fn and input distribution
                                [0, 1, 1], # change transition fn and input distribution
                                [1, 1, 1]]) # change reward fn, transition fn, and input distribution
        #change_matrix = np.array(list(product([0, 1], repeat=3)))
        change_matrix = change_matrix.astype(np.bool)
        self.tasks = self.env.unwrapped.unwrapped.random_tasks(change_matrix)

        # method 2: randomly generate tasks
        # total number of unique tasks in this class instance. note, the actual
        # environment (wrapped by this class) has many more task variations
        #num_tasks = 20
        #change_matrix = np.random.randint(low=0, high=2, size=(num_tasks, task_change_points))
        #change_matrix = change_matrix.astype(np.bool)
        #self.tasks = self.env.unwrapped.unwrapped.random_tasks(change_matrix)

        self.task_label_dim = len(self.tasks)
        self.one_hot = True
        if self.one_hot_labels:
            for idx in range(len(self.tasks)):
                label = np.zeros((self.task_label_dim,)).astype(np.float32)
                label[idx] = 1.
                self.tasks[idx]['task_label'] = label
                self.tasks[idx]['name'] = 'dynamic_grid_task_{0}'.format(idx + 1)
        else:
            labels = np.random.uniform(low=-1.,high=1.,size=(len(self.tasks), self.task_label_dim))
            labels = labels.astype(np.float32) 
            for idx in range(len(self.tasks)):
                self.tasks[idx]['task_label'] = labels[idx]
                self.tasks[idx]['name'] = 'dynamic_grid_task_{0}'.format(idx + 1)

        self.current_task = self.tasks[0]

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if done: state = self.reset()
        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        return state

    def reset_task(self, task_info):
        self.set_task(task_info)
        return self.reset()

    def set_task(self, task_info):
        msg = '`{0}` parameter should be included in `task_info` in the DynamicGrid env'
        assert 'goal_location' in task_info.keys(), msg.format('goal_location')
        assert 'transition_dynamics' in task_info.keys(), msg.format('transition_dynamics')
        assert 'permute_input' in task_info.keys(), msg.format('permute_input')
        self.env.unwrapped.unwrapped.set_task(task_info)
        self.current_task = task_info

    def get_task(self):
        return self.current_task

    def get_all_tasks(self, requires_task_label=True):
        if requires_task_label:
            tasks_label=np.random.uniform(low=-1.,high=1.,size=(len(self.tasks),self.task_label_dim))
            tasks = copy.deepcopy(self.tasks)
            for task, label in zip(tasks, tasks_label):
                task['task_label'] = label
            return tasks
        else:
            return self.tasks
    
    def random_tasks(self, num_tasks, requires_task_label=True):
        tasks_idx = np.random.randint(low=0, high=len(self.tasks), size=(num_tasks,))
        if requires_task_label:
            all_tasks = copy.deepcopy(self.tasks)
            tasks_label=np.random.uniform(low=-1.,high=1.,size=(len(self.tasks),self.task_label_dim))
            tasks = []
            for idx in tasks_idx:
                task = all_tasks[idx]
                task['task_label'] = tasks_label[idx]
                tasks.append(task)
            return tasks
        else:
            tasks = [self.tasks[idx] for idx in tasks_idx]
            return tasks

class DynamicGridFlatObs(DynamicGrid):
    # Dynamic Grid environment with flattend (1d vector) observations.
	# 2D images are flattened into 1D vectors
    def __init__(self, name, env_config_path=None, log_dir=None, seed=None, max_steps=100):
        super(DynamicGridFlatObs, self).__init__(name, env_config_path, log_dir, seed, max_steps)
        self.state_dim = int(np.prod(self.env.observation_space.shape))

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if done: state = self.reset()
        return state.ravel(), reward, done, info

    def reset(self):
        state = self.env.reset()
        return state.ravel()

class CTgraph(BaseTask):
    def __init__(self, name, env_config_path, log_dir=None):
        BaseTask.__init__(self)
        self.name = name
        import gym
        import gym_CTgraph
        env = gym.make(name, config_path=env_config_path)

        state = env.reset()
        #env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=state.shape,\
        #    dtype=np.float32)
        self.observation_space = env.observation_space
        self.action_dim = env.action_space.n
        if env.oneD:
            self.state_dim = int(np.prod(env.observation_space.shape))
        else:
            self.state_dim = env.observation_space.shape

        self.env = self.set_monitor(env, log_dir)

        depth = env.DEPTH
        branch = env.BRANCH

        # task label config
        self.task_label_dim = 2**depth
        self.one_hot_labels = True

        # get all tasks in graph environment instance
        from itertools import product
        tasks = list(product(list(range(branch)), repeat=depth))
        names = ['ctgraph_d{0}_b{1}_task_{2}'.format(depth, branch, idx) \
            for idx in range(len(tasks))] 
        self.tasks = [{'name': name, 'task': np.array(task), 'task_label': None} \
            for name, task in zip(names, tasks)]
        # generate label for each task
        if self.one_hot_labels:
            for idx in range(len(self.tasks)):
                label = np.zeros((self.task_label_dim,)).astype(np.float32)
                label[idx] = 1.
                self.tasks[idx]['task_label'] = label
        else:
            labels = np.random.uniform(low=-1.,high=1.,size=(len(self.tasks), self.task_label_dim))
            labels = labels.astype(np.float32) 
            for idx in range(len(self.tasks)):
                self.tasks[idx]['task_label'] = labels[idx]
        # set default task
        self.current_task = self.tasks[0]

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if done: state = self.reset()
        if self.env_config['image_dataset']['1D']: state = state.ravel()
        return state, reward, done, info

    def reset(self):
        ret = self.env.reset()
        state = ret
        if self.env_config['image_dataset']['1D']: state = state.ravel()
        return state

    def reset_task(self, taskinfo):
        self.set_task(taskinfo)
        return self.reset()

    def set_task(self, taskinfo):
        self.env.unwrapped.set_high_reward_path(taskinfo['task'])
        self.current_task = taskinfo
    
    def get_task(self):
        return self.current_task

    def get_all_tasks(self, requires_task_label=False):
        if requires_task_label:
            # one-hot labels
            #tasks_label = np.eye(len(self.tasks)).astype(np.float32)
            #tasks_label[tasks_label == 0.] = -1.
            # randomly sampled labels from uniform distribution
            tasks_label=np.random.uniform(low=-1.,high=1.,size=(len(self.tasks),self.task_label_dim))
            tasks_label = tasks_label.astype(np.float32)
            tasks = copy.deepcopy(self.tasks)
            for task, label in zip(tasks, tasks_label):
                task['task_label'] = label
            return tasks
        else:
            return self.tasks
    
    def random_tasks(self, num_tasks, requires_task_label=True):
        tasks_idx = np.random.randint(low=0, high=len(self.tasks), size=(num_tasks,))
        if requires_task_label:
            all_tasks = copy.deepcopy(self.tasks)
            # one-hot labels
            #tasks_label = np.eye(len(all_tasks)).astype(np.float32)
            #tasks_label[tasks_label == 0.] = -1.
            # randomly sampled labels from uniform distribution
            tasks_label=np.random.uniform(low=-1.,high=1.,size=(len(all_tasks),self.task_label_dim))
            tasks_label = tasks_label.astype(np.float32)
            tasks = []
            for idx in tasks_idx:
                task = all_tasks[idx]
                task['task_label'] = tasks_label[idx]
                tasks.append(task)
            return tasks
        else:
            tasks = [self.tasks[idx] for idx in tasks_idx]
            return tasks

class CTgraphFlatObs(CTgraph):
    # CTgraph environment with flattend (1d vector) observations.
    # observations are flattenend whether 1D or 2D observations.
    def __init__(self, name, env_config_path, log_dir=None):
        super(CTgraphFlatObs, self).__init__(name, env_config_path, log_dir)
        # overwrite previous written statedim to be flat 1d vector observations
        self.state_dim = int(np.prod(self.env.observation_space.shape))

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if done: state = self.reset()
        return state.ravel(), reward, done, info

    def reset(self):
        state = self.env.reset()
        return state.ravel()

class MetaCTgraph(BaseTask):
    '''
    ct-graph implementation where the class wrappers multiple ct-graph environments: each
    instance can vary in terms of depth (i.e., change in depth config)
    or in terms of observation distribution (image seed in config).

    note, only 2d state/observations are supported since in 1d states ct-graph, the state_dim
    can vary across different ct-graph depth.
    '''
    _TASK_ORDER_DEFAULT = 'default'
    _TASK_ORDER_INTERLEAVED = 'interleaved'
    _TASK_ORDER_RANDOM = 'random'
    _TASK_ORDER = ['default', 'interleaved', 'random']
    def __init__(self, name, env_config_path, log_dir=None):
        BaseTask.__init__(self)
        self.name = name

        # create all environment instances
        import os
        import gym
        import gym_CTgraph
        from itertools import product
        with open(env_config_path, 'r') as f:
            env_meta_config = json.load(f)
        task_order = env_meta_config['task_order']
        if task_order not in MetaCTgraph._TASK_ORDER:
            raise ValueError('`task_order` in config should be one of the following: {0}'\
                .format(MetaCTgraph._TASK_ORDER))
        base_path = os.path.dirname(env_config_path)
        envs = []
        for config_path in env_meta_config['config_paths']:
            env = gym.make('CTgraph-v0', config_path='{0}/{1}'.format(base_path, config_path))
            if env.oneD:
                raise ValueError('each environment should be configured to use 2d observation.'\
                    ' Set 1d config in {0}/{1} to false.'.format(base_path, config_path))
            envs.append(env)

        # observation/action space configuration
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        self.action_dim = envs[0].action_space.n
        self.state_dim = envs[0].observation_space.shape

        # generate tasks from all instantiated environments
        _all_tasks = []
        for idx, env in enumerate(envs):
            depth = env.DEPTH
            branch = env.BRANCH
            img_seed = env.conf_data['image_dataset']['seed']
            tasks = list(product(list(range(branch)), repeat=depth))
            names = ['ctgraph_d{0}_b{1}_imgseed_{2}_task_{3}'.format(depth, branch, img_seed, j) \
                for j in range(len(tasks))] 
            tasks = [{'name': name, 'task': np.array(task), 'task_label': None, 'env_idx': idx} \
                for name, task in zip(names, tasks)]
            _all_tasks.append(tasks)
        all_tasks = []
        if task_order == MetaCTgraph._TASK_ORDER_DEFAULT:
            for env_tasks in _all_tasks: all_tasks += env_tasks
        elif task_order == MetaCTgraph._TASK_ORDER_INTERLEAVED:
            if len(envs) == 1:
                raise ValueError('`interleaved` works when the number of envs (`config_paths`'\
                    ' in the config file) more than 1')
            for tasks_mix in itertools.zip_longest(*_all_tasks):
                for _task in tasks_mix:
                    if _task is not None: all_tasks.append(_task)
        elif task_order == MetaCTgraph._TASK_ORDER_RANDOM:
            for env_tasks in _all_tasks: all_tasks += env_tasks
            all_tasks = np.array(all_tasks)
            np.random.shuffle(all_tasks)
            all_tasks = all_tasks.tolist()
        del _all_tasks

        # set monitor
        envs = [self.set_monitor(env, log_dir) for env in envs]
        self.envs = envs
        self.tasks = all_tasks
        self.env = None

        # task label config
        self.task_label_dim = env_meta_config['label_dim']
        self.one_hot_labels = env_meta_config['one_hot']

        # generate label for each task
        if self.one_hot_labels:
            for idx in range(len(self.tasks)):
                label = np.zeros((self.task_label_dim,)).astype(np.float32)
                label[idx] = 1.
                self.tasks[idx]['task_label'] = label
        else:
            labels = np.random.uniform(low=-1.,high=1.,size=(len(self.tasks), self.task_label_dim))
            labels = labels.astype(np.float32) 
            for idx in range(len(self.tasks)):
                self.tasks[idx]['task_label'] = labels[idx]
        if 'filter_tasks' in env_meta_config.keys():
            filtered_tasks = []
            for idx_ in env_meta_config['filter_tasks']:
                filtered_tasks.append(self.tasks[idx_])
            self.tasks_ = self.tasks
            self.tasks = filtered_tasks
        # set default task
        self.set_task(self.tasks[0])

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if done: state = self.reset()
        if self.env.oneD: state = state.ravel()
        return state, reward, done, info

    def reset(self):
        ret = self.env.reset()
        state = ret
        if self.env.oneD: state = state.ravel()
        return state

    def reset_task(self, taskinfo):
        self.set_task(taskinfo)
        return self.reset()

    def set_task(self, taskinfo):
        self.env = self.envs[taskinfo['env_idx']]
        self.env.unwrapped.set_high_reward_path(taskinfo['task'])
        self.current_task = taskinfo

    def get_task(self):
        return self.current_task

    def get_all_tasks(self, requires_task_label=True):
        # `requires_task_label` left there for legacy/compatibility reasons to ensure uniformity
        # with other defined environments (e.g., minigrid and dynamic grid)
        return self.tasks

class MetaCTgraphFlatObs(MetaCTgraph):
    def __init__(self, name, env_config_path, log_dir=None):
        super(MetaCTgraphFlatObs, self).__init__(name, env_config_path, log_dir)
        # overwrite previous written statedim to be flat 1d vector observations
        self.state_dim = int(np.prod(self.env.observation_space.shape))

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if done: state = self.reset()
        return state.ravel(), reward, done, info

    def reset(self):
        state = self.env.reset()
        return state.ravel()

class MiniGrid(BaseTask):
    TIME_LIMIT=200
    def __init__(self, name, env_config_path, log_dir=None, seed=1000, eval_mode=False):
        BaseTask.__init__(self)
        self.name = name
        from gym.wrappers import TimeLimit
        import gym_minigrid
        from gym_minigrid.wrappers import ImgObsWrapper, ReseedWrapper, ActionBonus, StateBonus
        self.wrappers_dict = {'ActionBonus': ActionBonus, 'StateBonus': StateBonus}
        with open(env_config_path, 'r') as f:
            env_config = json.load(f)
        self.env_config = env_config
        env_names = env_config['tasks']
        if 'seeds' in env_config.keys():
            seeds = env_config['seeds']
        else:
            seeds = seed
            del seed

        if isinstance(seeds, int): seeds = [seeds,] * len(env_names)
        elif isinstance(seeds, list):
            assert len(seeds) == len(env_names), 'number of seeds in config file should match'\
                ' the number of tasks.'
        else: raise ValueError('invalid seed specification in config file')
        self.envs = {'{0}_seed{1}'.format(name, seed) : \
            ReseedWrapper(ImgObsWrapper(gym.make(name)), seeds=[seed,]) \
            for name, seed in zip(env_names, seeds)}
        env_names = ['{0}_seed{1}'.format(name, seed) for name, seed in zip(env_names, seeds)]
        #self.envs = {name: TimeLimit(env, MiniGrid.TIME_LIMIT) for name, env in self.envs.items()}

        # apply exploration bonus wrapper only to training envs
        if not eval_mode:
            if 'wrappers' in env_config.keys():
                for str_wrapper in env_config['wrappers']:
                    cls_wrapper = self.wrappers_dict[str_wrapper]
                    for k in self.envs.keys():
                        self.envs[k] = cls_wrapper(self.envs[k])
        self.observation_space = self.envs[env_names[0]].observation_space
        self.action_space = self.envs[env_names[0]].action_space
        self.state_dim = self.observation_space.shape
        # note, action_dim of 3 will reduce agent action to left, right, and forward
        if 'action_dim' in env_config.keys():
            self.action_dim = env_config['action_dim']
        else:
            self.action_dim = self.envs[env_names[0]].action_space.n
        # env monitors
        for name in self.envs.keys():
            self.envs[name] = self.set_monitor(self.envs[name], log_dir)
        # task label config
        self.task_label_dim = env_config['label_dim']
        self.one_hot_labels = True if env_config['one_hot'] else False
        # all tasks
        self.tasks = [{'name': name, 'task': name, 'task_label': None} \
            for name in self.envs.keys()]
        # generate label for each task
        if self.one_hot_labels:
            for idx in range(len(self.tasks)):
                label = np.zeros((self.task_label_dim,)).astype(np.float32)
                label[idx] = 1.
                self.tasks[idx]['task_label'] = label
        else:
            labels = np.random.uniform(low=-1.,high=1.,size=(len(self.tasks), self.task_label_dim))
            labels = labels.astype(np.float32) 
            for idx in range(len(self.tasks)):
                self.tasks[idx]['task_label'] = labels[idx]
        # set default task
        self.current_task = self.tasks[0]
        self.env = self.envs[self.current_task['task']]

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if done: state = self.reset()
        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        return state

    def reset_task(self, taskinfo):
        self.set_task(taskinfo)
        return self.reset()

    def set_task(self, taskinfo):
        self.current_task = taskinfo
        self.env = self.envs[self.current_task['task']]
    
    def get_task(self):
        return self.current_task

    def get_all_tasks(self, requires_task_label=True):
        return self.tasks
    
    def random_tasks(self, num_tasks, requires_task_label=True):
        raise NotImplementedError

class MiniGridFlatObs(MiniGrid):
    def __init__(self, name, env_config_path, log_dir=None, seed=1000, eval_mode=False):
        super(MiniGridFlatObs, self).__init__(name, env_config_path, log_dir, seed, eval_mode)
        self.state_dim = int(np.prod(self.env.observation_space.shape))

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if done: state = self.reset()
        return state.ravel(), reward, done, info

    def reset(self):
        state = self.env.reset()
        return state.ravel()

class ContinualWorld(BaseTask):

    # subtask: a configuration of object and goal polication in an env/task.
    RANDOMIZATION_STRATEGIES = [
        'deterministic', # subtask does not change in per reset in env/task.
        'random_init_all', # randomly generate a subtask per reset in env/task.
        'random_init_fixed20', # randomly selected out of 20 predefined subtask per reset in env/task
    ]
    def _env_instantiator(self, task_name, randomization):
        from gym.wrappers import TimeLimit
        from continualworld.utils.wrappers import RandomizationWrapper, SuccessCounter
        from continualworld.envs import get_subtasks, MT50, META_WORLD_TIME_HORIZON
        # adapted from get_single_env in continualworld codebase.
        env = MT50.train_classes[task_name]()
        env = RandomizationWrapper(env, get_subtasks(task_name), randomization)
        # Currently TimeLimit is needed since SuccessCounter looks at dones.
        #env = TimeLimit(env, META_WORLD_TIME_HORIZON)
        env = TimeLimit(env, 500)
        #env = SuccessCounter(env)
        env.name = task_name
        #env.num_envs = 1
        return env
        
    def __init__(self, name, env_config_path, log_dir=None, seed=1000):
        BaseTask.__init__(self)
        self.name = name

        with open(env_config_path, 'r') as f:
            env_config = json.load(f)
        self.env_config = env_config
        env_names = env_config['tasks']
        rconfig = env_config['randomization']
        if rconfig not in ContinualWorld.RANDOMIZATION_STRATEGIES:
            msg = '`randomization` in config should be on of the following: {0}'.format(\
                ContinualWorld.RANDOMIZATION_STRATEGIES)
            raise ValueError(msg)
        self.envs = {env_name : self._env_instantiator(env_name, rconfig) for env_name in env_names}
        self.observation_space = self.envs[env_names[0]].observation_space # (39,)
        self.action_space = self.envs[env_names[0]].action_space # (4,)
        self.state_dim = int(np.prod(self.observation_space.shape))
        self.action_dim = int(np.prod(self.action_space.shape))
        # env monitors
        for env_name in self.envs.keys():
            self.envs[env_name] = self.set_monitor(self.envs[env_name], log_dir)
        # task label config
        self.task_label_dim = env_config['label_dim']
        self.one_hot_labels = True if env_config['one_hot'] else False
        # all tasks
        self.tasks = [{'name': name, 'task': name, 'task_label': None} for name in self.envs.keys()]
        # generate label for each task
        if self.one_hot_labels:
            for idx in range(len(self.tasks)):
                label = np.zeros((self.task_label_dim,)).astype(np.float32)
                label[idx] = 1.
                self.tasks[idx]['task_label'] = label
        else:
            labels = np.random.uniform(low=-1.,high=1.,size=(len(self.tasks), self.task_label_dim))
            labels = labels.astype(np.float32) 
            for idx in range(len(self.tasks)):
                self.tasks[idx]['task_label'] = labels[idx]
        # set default task
        self.current_task = self.tasks[0]
        self.env = self.envs[self.current_task['task']]

    def step(self, action):
        _action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        state, reward, done, info = self.env.step(_action)
        if done: state = self.reset()
        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        return state

    def reset_task(self, taskinfo):
        self.set_task(taskinfo)
        return self.reset()

    def set_task(self, taskinfo):
        self.current_task = taskinfo
        self.env = self.envs[self.current_task['task']]
    
    def get_task(self):
        return self.current_task

    def get_all_tasks(self, requires_task_label=True):
        return self.tasks
    
    def random_tasks(self, num_tasks, requires_task_label=True):
        raise NotImplementedError

class PixelAtari(BaseTask):
    def __init__(self, name, seed=0, log_dir=None,
                 frame_skip=4, history_length=4, dataset=False):
        BaseTask.__init__(self)
        env = make_atari(name, frame_skip)
        env.seed(seed)
        if dataset:
            env = DatasetEnv(env)
            self.dataset_env = env
        env = self.set_monitor(env, log_dir)
        env = wrap_deepmind(env, history_length=history_length)
        self.env = env
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape
        self.name = name

class RamAtari(BaseTask):
    def __init__(self, name, no_op, frame_skip, log_dir=None):
        BaseTask.__init__(self)
        self.name = name
        env = gym.make(name)
        assert 'NoFrameskip' in env.spec.id
        env = self.set_monitor(env, log_dir)
        env = EpisodicLifeEnv(env)
        env = NoopResetEnv(env, noop_max=no_op)
        env = SkipEnv(env, skip=frame_skip)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        self.env = env
        self.action_dim = self.env.action_space.n
        self.state_dim = 128

class Pendulum(BaseTask):
    def __init__(self, log_dir=None):
        BaseTask.__init__(self)
        self.name = 'Pendulum-v0'
        self.env = gym.make(self.name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

    def step(self, action):
        return BaseTask.step(self, np.clip(2 * action, -2, 2))

class Box2DContinuous(BaseTask):
    def __init__(self, name, log_dir=None):
        BaseTask.__init__(self)
        self.name = name
        self.env = gym.make(self.name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

    def step(self, action):
        return BaseTask.step(self, np.clip(action, -1, 1))

class Roboschool(BaseTask):
    def __init__(self, name, log_dir=None):
        import roboschool
        BaseTask.__init__(self)
        self.name = name
        self.env = gym.make(self.name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

    def step(self, action):
        return BaseTask.step(self, np.clip(action, -1, 1))

class Bullet(BaseTask):
    def __init__(self, name, log_dir=None):
        import pybullet_envs
        BaseTask.__init__(self)
        self.name = name
        self.env = gym.make(name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

    def step(self, action):
        return BaseTask.step(self, np.clip(action, -1, 1))

class PixelBullet(BaseTask):
    def __init__(self, name, seed=0, log_dir=None, frame_skip=4, history_length=4):
        import pybullet_envs
        self.name = name
        env = gym.make(name)
        env.seed(seed)
        env = RenderEnv(env)
        env = self.set_monitor(env, log_dir)
        env = SkipEnv(env, skip=frame_skip)
        env = WarpFrame(env)
        env = WrapPyTorch(env)
        if history_length:
            env = StackFrame(env, history_length)
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape
        self.env = env

class ProcessTask:
    def __init__(self, task_fn, log_dir=None):
        self.pipe, worker_pipe = mp.Pipe()
        self.worker = ProcessWrapper(worker_pipe, task_fn, log_dir)
        self.worker.start()
        self.pipe.send([ProcessWrapper.SPECS, None])
        self.state_dim, self.action_dim, self.name = self.pipe.recv()

    def step(self, action):
        self.pipe.send([ProcessWrapper.STEP, action])
        return self.pipe.recv()

    def reset(self):
        self.pipe.send([ProcessWrapper.RESET, None])
        return self.pipe.recv()

    def close(self):
        self.pipe.send([ProcessWrapper.EXIT, None])

    def reset_task(self, task_info):
        self.pipe.send([ProcessWrapper.RESET_TASK, task_info])
        return self.pipe.recv()

    def set_task(self, task_info):
        self.pipe.send([ProcessWrapper.SET_TASK, task_info])

    def get_task(self):
        self.pipe.send([ProcessWrapper.GET_TASK, None])
        return self.pipe.recv()

    def get_all_tasks(self, requires_task_label):
        self.pipe.send([ProcessWrapper.GET_ALL_TASKS, requires_task_label])
        return self.pipe.recv()
    
    def random_tasks(self, num_tasks, requires_task_label):
        self.pipe.send([ProcessWrapper.RANDOM_TASKS, [num_tasks, requires_task_label]])
        return self.pipe.recv()

class ProcessWrapper(mp.Process):
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    RESET_TASK = 4
    SET_TASK = 5
    GET_TASK = 6
    GET_ALL_TASKS = 7
    RANDOM_TASKS = 8
    def __init__(self, pipe, task_fn, log_dir):
        mp.Process.__init__(self)
        self.pipe = pipe
        self.task_fn = task_fn
        self.log_dir = log_dir

    def run(self):
        np.random.seed()
        #seed = np.random.randint(0, sys.maxsize)
        seed = np.random.randint(0, 2**32 - 1)
        task = self.task_fn(log_dir=self.log_dir)
        task.seed(seed)
        while True:
            op, data = self.pipe.recv()
            if op == self.STEP:
                self.pipe.send(task.step(data))
            elif op == self.RESET:
                self.pipe.send(task.reset())
            elif op == self.EXIT:
                self.pipe.close()
                return
            elif op == self.SPECS:
                self.pipe.send([task.state_dim, task.action_dim, task.name])
            elif op == self.RESET_TASK:
                self.pipe.send(task.reset_task(data))
            elif op == self.SET_TASK:
                self.pipe.send(task.set_task(data))
            elif op == self.GET_TASK:
                self.pipe.send(task.get_task())
            elif op == self.GET_ALL_TASKS:
                self.pipe.send(task.get_all_tasks(data))
            elif op == self.RANDOM_TASKS:
                self.pipe.send(task.random_tasks(*data))
            else:
                raise Exception('Unknown command')

class ParallelizedTask:
    def __init__(self, task_fn, num_workers, log_dir=None, single_process=False):

        if single_process:
            self.tasks = [task_fn(log_dir=log_dir) for _ in range(num_workers)]
        else:
            self.tasks = [ProcessTask(task_fn, log_dir) for _ in range(num_workers)]
        self.state_dim = self.tasks[0].state_dim
        self.action_dim = self.tasks[0].action_dim
        self.name = self.tasks[0].name
        self.single_process = single_process

    def step(self, actions):
        results = [task.step(action) for task, action in zip(self.tasks, actions)]
        results = map(lambda x: np.stack(x), zip(*results))
        return results

    def reset(self):
        results = [task.reset() for task in self.tasks]
        return np.stack(results)

    def close(self):
        if self.single_process:
            return
        for task in self.tasks: task.close()

    def reset_task(self, task_info):
        results = [task.reset_task(task_info) for task in self.tasks]
        return np.stack(results)

    def set_task(self, task_info):
        for task in self.tasks:
            task.set_task(task_info)

    def get_task(self, all_workers=False):
        if not all_workers:
            return self.tasks[0].get_task()
        else:
            return [task.get_task() for task in self.tasks]

    def get_all_tasks(self, requires_task_label):
        return self.tasks[0].get_all_tasks(requires_task_label)
    
    def random_tasks(self, num_tasks, requires_task_label):
        return self.tasks[0].random_tasks(num_tasks, requires_task_label)
