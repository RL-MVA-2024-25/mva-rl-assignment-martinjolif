from copyreg import pickle
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from utils import greedy_action_FQI

import gzip

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def act(self, observation, use_random=False):
        action = greedy_action_FQI(self.QfunctionsLoaded, observation, env.action_space.n)
        return action

    def save(self, path):
        pass

    def load(self):
        with gzip.open('src/FQI_Q_functions.pkl.gz', 'rb') as f:
            self.QfunctionsLoaded = pickle.load(f)
