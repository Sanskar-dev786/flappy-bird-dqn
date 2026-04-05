import itertools
import random
import flappy_bird_gymnasium
import torch
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory

import yaml

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class Agent: 
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)

        hyperparameters = all_hyperparameter_sets.get(hyperparameter_set)

        if hyperparameters is None:
            raise ValueError(f"'{hyperparameter_set}' not found in YAML file")

        # Load parameters
        self.env_id = hyperparameters['env_id']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']


    def run(self, is_training=True, render=False):
      
    #  env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)
        env = gym.make(self.env_id, render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        rewards_per_episode = []
        epsilon_history = []
        
        policy_dqn = DQN(num_states, num_actions,).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)
            terminated = False
            episode_reward = 0.0




            while not terminated:

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        # tensor([1, 2, 3, ......]) ==> tensor([[1, 2, 3, ....]])
                        action = policy_dqn(state).argmax()


                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())
               
                # accumalated reward
                episode_reward += reward


                #convet the new state and reward to the tensor
                new_state = torch.tensor(new_state, dtype= torch.float, device= device)
                reward = torch.tensor(reward, dtype=torch.float, device= device)
                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                
                # Move to the next state
                state = new_state

            rewards_per_episode.append(episode_reward)

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

if __name__ == '__main__':
    agent = Agent("flappybird")
    agent.run(is_training=True, render=True)

