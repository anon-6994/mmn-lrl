#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from copy import deepcopy
from ...network import *
from ...component import *
from ..BaseAgent import *

class A2CAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()
        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        for _ in range(config.rollout_length):
            actions, log_probs, entropy, values = self.network.predict(config.state_normalizer(states))
            next_states, rewards, terminals, _ = self.task.step(actions.detach().cpu().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0

            rollout.append([log_probs, values, actions, rewards, 1 - terminals, entropy])
            states = next_states

        self.states = states
        pending_value = self.network.predict(config.state_normalizer(states))[-1]
        rollout.append([None, pending_value, None, None, None, None])

        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            log_prob, value, actions, rewards, terminals, entropy = rollout[i]
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount * terminals * next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [log_prob, value, returns, advantages, entropy]

        log_prob, value, returns, advantages, entropy = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        policy_loss = -log_prob * advantages
        value_loss = 0.5 * (returns - value).pow(2)
        entropy_loss = entropy.mean()

        self.policy_loss = np.mean(policy_loss.cpu().detach().numpy())
        self.entropy_loss = np.mean(entropy_loss.cpu().detach().numpy())
        self.value_loss = np.mean(value_loss.cpu().detach().numpy())

        self.optimizer.zero_grad()
        (policy_loss - config.entropy_weight * entropy_loss +
         config.value_loss_weight * value_loss).mean().backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps

class A2CContinualLearnerAgent(BaseContinualLearnerAgent):
    def __init__(self, config):
        BaseContinualLearnerAgent.__init__(self, config)
        self.config = config
        self.task = None if config.task_fn is None else config.task_fn()
        if config.eval_task_fn is None:
            self.evaluation_env = None
        else:
            self.evaluation_env = config.eval_task_fn(config.log_dir)
            self.task = self.evaluation_env if self.task is None else self.task
        tasks_info = self.task.get_all_tasks(config.cl_requires_task_label)[ : config.cl_num_tasks]

        # NOTE, to remove
        #tasks_info = tasks_info + tasks_info + tasks_info
        #config.cl_num_tasks = len(tasks_info)
        # shuffle tasks
        #idxs = [0, 3, 1, 2]
        #idxs = [0, 1, 2, 3]
        #tasks_info = [tasks_info[i] for i in idxs]

        self.config.cl_tasks_info = tasks_info
        label_dim = 0 if tasks_info[0]['task_label'] is None else len(tasks_info[0]['task_label'])

        self.network = config.network_fn(self.task.state_dim, self.task.action_dim, label_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters(), config.lr)
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)

        self.data_buffer = []
        # weight preservation setup
        self.params = {n: p for n, p in self.network.named_parameters() if p.requires_grad}
        self.precision_matrices = {}
        self.means = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            self.precision_matrices[n] = p.data.to(config.DEVICE)
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            self.means[n] = p.data.to(config.DEVICE)

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        current_task_label = self.task.get_task()['task_label']
        batch_dim = len(states) # same as config.num_workers
        if batch_dim == 1:
            current_task_label = current_task_label.reshape(1, -1)
        else:
            current_task_label = np.repeat(current_task_label.reshape(1, -1), batch_dim, axis=0)
        if config.cl_preservation != 'ewc': self.data_buffer.append(states)
        for _ in range(config.rollout_length):
            _, actions, log_probs, entropy, values = self.network.predict(states, \
                task_label=current_task_label)
            next_states, rewards, terminals, _ = self.task.step(actions.detach().cpu().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0

            rollout.append([log_probs, values, actions, rewards, 1 - terminals, entropy])
            next_states = config.state_normalizer(next_states)
            states = next_states
            if config.cl_preservation != 'ewc': self.data_buffer.append(states)

        self.states = states
        pending_value = self.network.predict(states, task_label=current_task_label)[-1]
        rollout.append([None, pending_value, None, None, None, None])

        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            log_prob, value, actions, rewards, terminals, entropy = rollout[i]
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error=rewards + config.discount * terminals * next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [log_prob, value, returns, advantages, entropy]

        log_prob, value, returns, advantages, entropy = map(lambda x: torch.cat(x, dim=0), \
            zip(*processed_rollout))
        policy_loss = -log_prob * advantages
        value_loss = 0.5 * (returns - value).pow(2)
        entropy_loss = entropy.mean()
        weight_pres_loss = self.penalty()

        self.policy_loss = np.mean(policy_loss.cpu().detach().numpy())
        self.entropy_loss = np.mean(entropy_loss.cpu().detach().numpy())
        self.value_loss = np.mean(value_loss.cpu().detach().numpy())

        self.optimizer.zero_grad()
        (policy_loss - config.entropy_weight * entropy_loss +
         config.value_loss_weight * value_loss + weight_pres_loss).mean().backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps

    def penalty(self):
        loss = 0
        for n, p in self.network.named_parameters():
            _loss = self.precision_matrices[n] * (p - self.means[n]) ** 2
            loss += _loss.sum()
        return loss * self.config.cl_loss_coeff

    def consolidate(self, batch_size=32):
        raise NotImplementedError

class A2CAgentBaseline(A2CContinualLearnerAgent):
    '''
    A2C continual learning agent without preservation/consolidation
    '''
    def __init__(self, config):
        A2CContinualLearnerAgent.__init__(self, config)

    def consolidate(self, batch_size=32):
        print('sanity check, calling consolidate in agent w/o preservation')
        # NOTE, the return values are zeros and do no consolidate any weights
        # therefore, all parameters are retrained/finetuned per task.
        return self.precision_matrices, self.precision_matrices

class A2CAgentMAS(A2CContinualLearnerAgent):
    '''
    A2C continual learning agent using memory aware synapse (MAS)
    weight preservation mechanism
    '''
    def __init__(self, config):
        A2CContinualLearnerAgent.__init__(self, config)

    def consolidate(self, batch_size=32):
        states = np.concatenate(self.data_buffer)
        task_label = self.task.get_task()['task_label']
        task_label = np.repeat(task_label.reshape(1, -1), len(states), axis=0)

        states = states[-2000 : ]
        task_label = task_label[-2000 : ]

        config = self.config
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(config.DEVICE)

        # Set the model in the evaluation mode
        self.network.eval()

        # Get network outputs
        idxs = np.arange(len(states))
        np.random.shuffle(idxs)
        # shuffle data
        states = states[idxs]
        task_label = task_label[idxs]

        num_batches = len(states) // batch_size
        num_batches = num_batches + 1 if len(states) % batch_size > 0 else num_batches
        for batch_idx in range(num_batches):
            start, end = batch_idx * batch_size, (batch_idx+1) * batch_size
            states_ = states[start:end, ...]
            task_label_ = task_label[start:end, ...]
            logits, actions, _, _, values = self.network.predict(states_, task_label=task_label_)
            logits = torch.softmax(logits, dim=1)
            # get value loss
            value_loss = values.sum()
            # get actor loss: l2 norm of each row of logits
            try:
                actor_loss = (torch.linalg.norm(logits, ord=2, dim=1)).sum()
            except:
                # older version of pytorch, we calculate l2 norm as API is not available
                actor_loss = (logits ** 2).sum(dim=1).sqrt().sum()
            #loss = actor_loss
            #loss = actor_loss + value_loss
            #self.network.zero_grad()
            #loss.backward()
            ## Update the temporary precision matrix
            #for n, p in self.params.items():
            #    precision_matrices[n].data += p.grad.data ** 2 / float(len(states))
            #    #precision_matrices[n].data += p.grad.data ** 2
            self.network.zero_grad()
            actor_loss.backward()
            # Update the temporary precision matrix
            for n, p in self.params.items():
                #precision_matrices[n].data += p.grad.data ** 2 / float(len(states))
                precision_matrices[n].data += p.grad.data ** 2
            self.network.zero_grad()
            value_loss.backward()
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

class A2CAgentSCP(A2CContinualLearnerAgent):
    '''
    A2C continual learning agent using sliced cramer preservation (SCP)
    weight preservation mechanism
    '''
    def __init__(self, config):
        A2CContinualLearnerAgent.__init__(self, config)

    def consolidate(self, batch_size=32):
        states = np.concatenate(self.data_buffer)
        task_label = self.task.get_task()['task_label']
        task_label = np.repeat(task_label.reshape(1, -1), len(states), axis=0)
        config = self.config
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(config.DEVICE)

        # Set the model in the evaluation mode
        self.network.eval()

        # Get network outputs
        idxs = np.arange(len(states))
        np.random.shuffle(idxs)
        # shuffle data
        states = states[idxs]
        task_label = task_label[idxs]

        num_batches = len(states) // batch_size
        num_batches = num_batches + 1 if len(states) % batch_size > 0 else num_batches
        for batch_idx in range(num_batches):
            start, end = batch_idx * batch_size, (batch_idx+1) * batch_size
            states_ = states[start:end, ...]
            task_label_ = task_label[start:end, ...]
            self.network.zero_grad()
            logits, actions, _, _, values = self.network.predict(states_, task_label=task_label_)
            logits_mean = logits.type(torch.float32).mean(dim=0)
            K = logits_mean.shape[0]
            values_mean = values.type(torch.float32).mean(dim=0)
            L = values_mean.shape[0]
            for _ in range(config.cl_n_slices):
                xi = torch.randn(K, ).to(config.DEVICE)
                xi /= torch.sqrt((xi**2).sum())
                vi = torch.randn(L, ).to(config.DEVICE)
                vi /= torch.sqrt((vi**2).sum())
                out_actor = torch.matmul(logits_mean, xi)
                out_value = torch.matmul(values_mean, vi) 
                out = out_actor + out_value
                self.network.zero_grad()
                out.backward(retain_graph=True)
                # Update the temporary precision matrix
                for n, p in self.params.items():
                    precision_matrices[n].data += p.grad.data ** 2 / float(len(states))

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

class A2CAgentEWC(A2CContinualLearnerAgent):
    '''
    A2C continual learning agent using elastic weight consolidation (EWC)
    weight preservation mechanism
    '''
    def __init__(self, config):
        A2CContinualLearnerAgent.__init__(self, config)

    def consolidate(self, batch_size=16):
        print('sanity check. consolidation in ewc')
        config = self.config
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(config.DEVICE)

        # Set the model in the evaluation mode
        self.network.eval()
        # collect data and consolidate
        self.states = config.state_normalizer(self.task.reset())
        for _ in range(batch_size):
            rollout = []
            states = self.states
            current_task_label = self.task.get_task()['task_label']
            batch_dim = len(states) # same as config.num_workers
            if batch_dim == 1:
                current_task_label = current_task_label.reshape(1, -1)
            else:
                current_task_label = np.repeat(current_task_label.reshape(1, -1), batch_dim, axis=0)
            for _ in range(config.rollout_length):
                _, actions, log_probs, entropy, values = self.network.predict(states, \
                    task_label=current_task_label)
                next_states, rewards, terminals, _ = self.task.step(actions.detach().cpu().numpy())
                self.episode_rewards += rewards
                rewards = config.reward_normalizer(rewards)
                for i, terminal in enumerate(terminals):
                    if terminals[i]:
                        self.last_episode_rewards[i] = self.episode_rewards[i]
                        self.episode_rewards[i] = 0

                rollout.append([log_probs, values, actions, rewards, 1 - terminals, entropy])
                next_states = config.state_normalizer(next_states)
                states = next_states

            self.states = states
            pending_value = self.network.predict(states, task_label=current_task_label)[-1]
            rollout.append([None, pending_value, None, None, None, None])

            processed_rollout = [None] * (len(rollout) - 1)
            advantages = tensor(np.zeros((config.num_workers, 1)))
            returns = pending_value.detach()
            for i in reversed(range(len(rollout) - 1)):
                log_prob, value, actions, rewards, terminals, entropy = rollout[i]
                terminals = tensor(terminals).unsqueeze(1)
                rewards = tensor(rewards).unsqueeze(1)
                next_value = rollout[i + 1][1]
                returns = rewards + config.discount * terminals * returns
                if not config.use_gae:
                    advantages = returns - value.detach()
                else:
                    td_error=rewards + config.discount*terminals*next_value.detach() - value.detach()
                    advantages = advantages * config.gae_tau * config.discount * terminals + td_error
                processed_rollout[i] = [log_prob, value, returns, advantages, entropy]

            log_prob, value, returns, advantages, entropy = map(lambda x: torch.cat(x, dim=0), \
                zip(*processed_rollout))
            policy_loss = -log_prob * advantages
            value_loss = 0.5 * (returns - value).pow(2)
            entropy_loss = entropy.mean()

            self.network.zero_grad()
            loss = (policy_loss - config.entropy_weight * entropy_loss +
                config.value_loss_weight * value_loss).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
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

class A2CAgentNM(A2CContinualLearnerAgent):
    '''
    A2C continual learning agent without preservation/consolidation
    importance paramters for consolidation are generated by a neuromodulated (hyper) network
    '''
    def __init__(self, config):
        A2CContinualLearnerAgent.__init__(self, config)

    def consolidate(self, batch_size=32):
        print('sanity check, calling consolidate in agent w/o preservation')
        # NOTE, the return values are zeros and do no consolidate any weights
        # therefore, all parameters are retrained/finetuned per task.
        return self.precision_matrices, self.precision_matrices
