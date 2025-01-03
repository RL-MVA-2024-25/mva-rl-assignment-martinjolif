import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
from utils import ReplayBuffer
from utils import greedy_action_DQN
from src.evaluate import evaluate_HIV


class dqn_agent:
    def __init__(self, config, model):
        self.device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'], self.device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop
        self.total_steps = 0
        self.model = model
        if config['use_Huber_loss']:
            self.criterion = torch.nn.SmoothL1Loss()
        else:
            self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.nb_gradient_steps = config['gradient_steps']
        self.target_model = deepcopy(self.model).to(self.device)
        self.update_target_strategy = config['update_target_strategy']
        self.update_target_freq = config['update_target_freq']
        self.update_target_tau = config['update_target_tau']

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma)
            #update = torch.addcmul(R, self.gamma, 1 - D, QYmax)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best_test_reward = 0

        with tqdm(total=max_episode, desc="Episode Progress") as pbar:
            while episode < max_episode:
                # update epsilon
                if step > self.epsilon_delay:
                    epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)

                # select epsilon-greedy action
                if np.random.rand() < epsilon:
                    action = np.random.randint(self.nb_actions)
                else:
                    action = greedy_action_DQN(self.model, state)

                # step
                next_state, reward, done, trunc, _ = env.step(action)
                self.memory.append(state, action, reward, next_state, done)
                episode_cum_reward += reward

                # train
                for _ in range(self.nb_gradient_steps):
                    self.gradient_step()

                # update target network if needed
                if self.update_target_strategy == 'replace':
                    if step % self.update_target_freq == 0:
                        self.target_model.load_state_dict(self.model.state_dict())
                if self.update_target_strategy == 'ema':
                    target_state_dict = self.target_model.state_dict()
                    model_state_dict = self.model.state_dict()
                    tau = self.update_target_tau
                    for key in model_state_dict:
                        target_state_dict[key] = tau * model_state_dict + (1 - tau) * target_state_dict
                    self.target_model.load_state_dict(target_state_dict)


                # next transition
                step += 1
                if done or trunc:
                    #save the best model
                    test_reward = episode_cum_reward
                    if test_reward > best_test_reward:
                        best_test_reward = test_reward
                        self.best_model = deepcopy(self.model).to(self.device)

                    episode += 1
                    pbar.set_postfix({
                        "Episode": episode,
                        "Epsilon": epsilon,
                        "Batch Size": len(self.memory),
                        "Episode Return": episode_cum_reward
                    })
                    pbar.update(1)
                    state, _ = env.reset()
                    episode_return.append(episode_cum_reward)
                    episode_cum_reward = 0
                else:
                    state = next_state

        torch.save(self.best_model.state_dict(), 'checkpoint-DQN.pth')
        return episode_return