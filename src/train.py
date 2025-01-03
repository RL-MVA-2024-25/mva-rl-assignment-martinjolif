from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from agents.utils import greedy_action_DQN
import torch.nn as nn
import torch

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def act(self, observation, use_random=False):
        #For FQI
        """
        action = greedy_action_FQI(self.QfunctionsLoaded, observation, env.action_space.n)
        """
        #For DQN
        action = greedy_action_DQN(self.DQN_model, observation)

        return action

    def save(self, path):
        pass

    def load(self):
        #For loading FQI
        """
        with gzip.open('src/agents/FQI_Q_functions-2-9.pkl.gz', 'rb') as f:
            self.QfunctionsLoaded = pickle.load(f)
        """

        #For loading DQN
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.DQN_model = torch.nn.Sequential(nn.Linear(env.observation_space.shape[0], 256),
                          nn.ReLU(),
                          nn.Linear(256, 128),
                          nn.ReLU(),
                          nn.Linear(128, 256),
                          nn.ReLU(),
                          nn.Linear(256, env.action_space.n)).to(device)
        self.DQN_model.load_state_dict(torch.load('src/agents/checkpoint-DQN-7-9.pth', map_location=device, weights_only=False))
        self.DQN_model.eval()