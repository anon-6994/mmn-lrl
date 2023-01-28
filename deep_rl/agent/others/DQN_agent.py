#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ...network import *
from ...component import *
from ...utils import *
import time
from ..BaseAgent import *
from copy import deepcopy

class DQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.criterion = nn.MSELoss()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.policy = config.policy_fn()
        self.total_steps = 0

    def episode(self, deterministic=False):
        episode_start_time = time.time()
        #print 'begin reset'
        state = self.task.reset()
        total_reward = 0.0
        steps = 0
        #print 'begin episode'
        while True:
            value = self.network.predict(np.stack([self.config.state_normalizer(state)]), True).flatten()
            if deterministic:
                action = np.argmax(value)
            elif self.total_steps < self.config.exploration_steps:
                action = np.random.randint(0, len(value))
            else:
                action = self.policy.sample(value)
            #print 'call self.task.step(action)'
            next_state, reward, done, _ = self.task.step(action)
            #print 'task step'
            total_reward += reward
            reward = self.config.reward_normalizer(reward)
            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done)])
                self.total_steps += 1
            steps += 1
            state = next_state
            #print 'learning'
            if not deterministic and self.total_steps > self.config.exploration_steps \
                    and self.total_steps % self.config.sgd_update_frequency == 0:
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences
                states = self.config.state_normalizer(states)
                next_states = self.config.state_normalizer(next_states)
                q_next = self.target_network.predict(next_states, False).detach()
                if self.config.double_q:
                    _, best_actions = self.network.predict(next_states).detach().max(1)
                    q_next = q_next.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                else:
                    q_next, _ = q_next.max(1)
                terminals = tensor(terminals)
                rewards = tensor(rewards)
                q_next = self.config.discount * q_next * (1 - terminals)
                q_next.add_(rewards)
                actions = tensor(actions).unsqueeze(1).long()
                q = self.network.predict(states, False)
                q = q.gather(1, actions).squeeze(1)
                loss = self.criterion(q, q_next)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            #print 'self evaluate'
            self.evaluate()
            if not deterministic and self.total_steps % self.config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())
            if not deterministic and self.total_steps > self.config.exploration_steps:
                self.policy.update_epsilon()
            #print 'chekc is done'
            if done:
                #print 'end eposide'
                break

        episode_time = time.time() - episode_start_time
        self.config.logger.debug('episode steps %d, episode time %f, time per step %f' %
                          (steps, episode_time, episode_time / float(steps)))

        return total_reward, steps

class DQNContinualLearnerAgent(BaseContinualLearnerAgent):
    def __init__(self, config):
        BaseContinualLearnerAgent.__init__(self, config)

        self.config = config
        self.task = None if config.task_fn is None else config.task_fn(config.log_dir)
        if config.eval_task_fn is None:
            self.evaluation_env = None
        else:
            self.evaluation_env = config.eval_task_fn(config.log_dir)
            self.task = self.evaluation_env if self.task is None else self.task
        tasks_info = self.task.get_all_tasks(config.cl_requires_task_label)[ : config.cl_num_tasks]

        self.config.cl_tasks_info = tasks_info
        label_dim = 0 if tasks_info[0]['task_label'] is None else len(tasks_info[0]['task_label'])

        self.network = config.network_fn(self.task.state_dim, self.task.action_dim, label_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim, label_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters(), config.lr)
        self.criterion = nn.MSELoss()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.policy = config.policy_fn()
        self.total_steps = 0

        # continual learning (cl) setup
        self.params = {n: p for n, p in self.network.named_parameters() if p.requires_grad}
        self.precision_matrices = {}
        self.means = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            self.precision_matrices[n] = p.data.to(config.DEVICE)
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            self.means[n] = p.data.to(config.DEVICE)

    def episode(self, deterministic=False):
        episode_start_time = time.time()
        state = self.task.reset()
        total_reward = 0.0
        steps = 0
        current_task_label = self.task.get_task()['task_label']
        current_task_label = np.stack([current_task_label,])
        while True:
            value = self.network.predict(np.stack([self.config.state_normalizer(state)]), \
                current_task_label, True).flatten()
            if deterministic:
                action = np.argmax(value)
            elif self.total_steps < self.config.exploration_steps:
                action = np.random.randint(0, len(value))
            else:
                action = self.policy.sample(value)
            next_state, reward, done, _ = self.task.step(action)
            total_reward += reward
            reward = self.config.reward_normalizer(reward)
            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done), \
                    current_task_label.ravel()])
                self.total_steps += 1
            steps += 1
            state = next_state
            if not deterministic and self.total_steps > self.config.exploration_steps \
                    and self.total_steps % self.config.sgd_update_frequency == 0:
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals, task_labels = experiences
                states = self.config.state_normalizer(states)
                next_states = self.config.state_normalizer(next_states)
                q_next = self.target_network.predict(next_states, task_labels, False).detach()
                if self.config.double_q:
                    _, best_actions = self.network.predict(next_states, task_labels).detach().max(1)
                    q_next = q_next.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                else:
                    q_next, _ = q_next.max(1)
                terminals = tensor(terminals)
                rewards = tensor(rewards)
                q_next = self.config.discount * q_next * (1 - terminals)
                q_next.add_(rewards)
                actions = tensor(actions).unsqueeze(1).long()
                q = self.network.predict(states, task_labels, False)
                q = q.gather(1, actions).squeeze(1)
                loss = self.criterion(q, q_next)
                weight_pres_loss = self.penalty()
                loss = loss + weight_pres_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            self.evaluate()
            if not deterministic and self.total_steps % self.config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())
            if not deterministic and self.total_steps > self.config.exploration_steps:
                self.policy.update_epsilon()
            if done:
                break

        episode_time = time.time() - episode_start_time
        self.config.logger.debug('episode steps %d, episode time %f, time per step %f' %
                          (steps, episode_time, episode_time / float(steps)))

        return total_reward, steps

    def penalty(self):
        loss = 0
        for n, p in self.network.named_parameters():
            _loss = self.precision_matrices[n] * (p - self.means[n]) ** 2
            loss += _loss.sum()
        return loss * self.config.cl_loss_coeff

    def consolidate(self, num_itr=32):
        raise NotImplementedError

class DQNAgentBaseline(DQNContinualLearnerAgent):
    def __init__(self, config):
        DQNContinualLearnerAgent.__init__(self, config)

    def consolidate(self, num_itr=32):
        print('sanity check, consolidate in baseline dqn agent')
        return self.precision_matrices, self.precision_matrices

class DQNAgentMAS(DQNContinualLearnerAgent):
    def __init__(self, config):
        DQNContinualLearnerAgent.__init__(self, config)

    def consolidate(self, num_itr=32):
        print('sanity check, consolidate in mas dqn agent')
        config = self.config
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(config.DEVICE)

        self.network.eval()
        for _ in range(num_itr):
            experiences = self.replay.sample()
            states, _, _, _, _, task_labels = experiences
            states = self.config.state_normalizer(states)
            q_values = self.network.predict(states, task_labels, False)
            try:
                output = (torch.linalg.norm(q_values, ord=2, dim=1)).sum()
            except:
                # older version of pytorch, we calculate l2 norm as API is not available
                output = (q_values ** 2).sum(dim=1).sqrt().sum()
            self.network.zero_grad()
            output.backward()
            # Update the temporary precision matrix
            for n, p in self.params.items():
                #precision_matrices[n].data += p.grad.data ** 2 / float(len(states))
                precision_matrices[n].data += p.grad.data ** 2

        for n, p in self.network.named_parameters():
            if p.requires_grad is False: continue
            # Update the precision matrix
            self.precision_matrices[n] = config.cl_alpha*self.precision_matrices[n] + \
                (1 - config.cl_alpha) * precision_matrices[n]
            # Update the means
            self.means[n] = deepcopy(p.data).to(config.DEVICE)

        self.network.train()
        # return task precision matrices and general precision matrices across tasks agent has
        # been explosed to so far
        return precision_matrices, self.precision_matrices

class DQNAgentSCP(DQNContinualLearnerAgent):
    def __init__(self, config):
        DQNContinualLearnerAgent.__init__(self, config)

    def consolidate(self, num_itr=32):
        print('sanity check, consolidate in scp dqn agent')
        config = self.config
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(config.DEVICE)

        self.network.eval()
        for _ in range(num_itr):
            experiences = self.replay.sample()
            states, _, _, _, _, task_labels = experiences
            states = self.config.state_normalizer(states)

            self.network.zero_grad()
            q_values = self.network.predict(states, task_labels, False)
            q_values_mean = q_values.type(torch.float32).mean(dim=0)
            K = q_values_mean.shape[0]
            for _ in range(config.cl_n_slices):
                xi = torch.randn(K, ).to(config.DEVICE)
                xi /= torch.sqrt((xi**2).sum())
                out = torch.matmul(q_values_mean, xi)
                self.network.zero_grad()
                out.backward(retain_graph=True)
                # Update the temporary precision matrix
                for n, p in self.params.items():
                    #precision_matrices[n].data += p.grad.data ** 2 / float(len(states))
                    precision_matrices[n].data += p.grad.data ** 2

        for n, p in self.network.named_parameters():
            if p.requires_grad is False: continue
            # Update the precision matrix
            self.precision_matrices[n] = config.cl_alpha*self.precision_matrices[n] + \
                (1 - config.cl_alpha) * precision_matrices[n]
            # Update the means
            self.means[n] = deepcopy(p.data).to(config.DEVICE)

        self.network.train()
        # return task precision matrices and general precision matrices across tasks agent has
        # been explosed to so far
        return precision_matrices, self.precision_matrices

class DQNAgentEWC(DQNContinualLearnerAgent):
    def __init__(self, config):
        DQNContinualLearnerAgent.__init__(self, config)

    def consolidate(self, num_itr=32):
        config = self.config
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(config.DEVICE)

        self.network.eval()
        for _ in range(num_itr):
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals, task_labels = experiences
            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)
            q_next = self.target_network.predict(next_states, task_labels, False).detach()
            if self.config.double_q:
                _, best_actions = self.network.predict(next_states, task_labels).detach().max(1)
                q_next = q_next.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            else:
                q_next, _ = q_next.max(1)
            terminals = tensor(terminals)
            rewards = tensor(rewards)
            q_next = self.config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)
            actions = tensor(actions).unsqueeze(1).long()
            q = self.network.predict(states, task_labels, False)
            q = q.gather(1, actions).squeeze(1)
            loss = self.criterion(q, q_next)
            self.network.zero_grad()
            loss.backward()
            # Update the temporary precision matrix
            for n, p in self.params.items():
                #precision_matrices[n].data += p.grad.data ** 2 / float(len(states))
                precision_matrices[n].data += p.grad.data ** 2

        for n, p in self.network.named_parameters():
            if p.requires_grad is False: continue
            # Update the precision matrix
            self.precision_matrices[n] = config.cl_alpha*self.precision_matrices[n] + \
                (1 - config.cl_alpha) * precision_matrices[n]
            # Update the means
            self.means[n] = deepcopy(p.data).to(config.DEVICE)

        self.network.train()
        # return task precision matrices and general precision matrices across tasks agent has
        # been explosed to so far
        return precision_matrices, self.precision_matrices
