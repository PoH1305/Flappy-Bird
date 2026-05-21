import gymnasium as gym
import flappy_bird_gymnasium
import torch
import torch.nn as nn
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random
import torch.optim as optim
import os
import argparse

RUN_DIR = "runs"
os.makedirs(RUN_DIR, exist_ok=True)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")


class Agent:

    def __init__(self, parame_set):

        self.parame_set = parame_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[parame_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.min_batch_size = params["min_batch_size"]
        self.replay_memory_size = params["replay_memory_size"]
        self.network_sync_rate = params["network_sync_rate"]

        self.reward_threshold = params["reward_threshold"]
        self.min_batch_size = params["min_batch_size"]

        self.LOG_FILES = os.path.join(RUN_DIR, f"{self.parame_set}.log")
        self.MODEL_FILES = os.path.join(RUN_DIR, f"{self.parame_set}.pt")        
        self.loss = nn.MSELoss()
        self.optimizer = None
    
    def run(self, is_training = True, render = False):

        env = gym.make("FlappyBird-v0", render_mode = "human" if render else None)

        num_state = env.observation_space.shape[0]
        num_action = env.action_space.n

        policy_dqn = DQN(num_state, num_action).to(device)


        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_state, num_action).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            step = 0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr = self.alpha)

            best_reward = float("-inf")

        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILES))
            policy_dqn.eval()


        for episode in itertools.count():
            
            state, _ = env.reset()
            state = torch.tensor(state, dtype = torch.float32, device = device)

            episode_reward = 0
            terminated = False
            
            while (not terminated and episode_reward < self.reward_threshold):

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype = torch.long, device = device)    # explore
                else:
                    with torch.no_grad():
                        action =policy_dqn(state.unsqueeze(dim = 0)).squeeze().argmax()    # exploit

                next_state, reward, terminated, _, _ = env.step(action.item())

                reward = torch.tensor(reward, dtype = torch.float32, device = device)
                next_state = torch.tensor(next_state, dtype = torch.float32, device = device)

                if is_training:
                    memory.push((state, action, next_state, reward, terminated))
                    step += 1

                state = next_state

                episode_reward += reward.item()

            print(f" episode {episode+1} : total reward is {episode_reward}")


            if is_training:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                
                if episode_reward > best_reward:
                    log_msg = f" best reward = {episode_reward} at episode {episode+1}"

                    with open(self.LOG_FILES, 'a') as f:
                        f.write(log_msg + "\n")
                    
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILES)
                    best_reward = episode_reward


            if is_training and len(memory) >= self.min_batch_size:
                min_batch = memory.sample(self.min_batch_size)

                self.optimize(min_batch, policy_dqn, target_dqn)

                #Sync network

                if step > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step = 0
    
    def optimize(self, min_batch, policy_dqn, target_dqn):
        
        state, action, next_state, reward, terminated = zip(*min_batch)

        state = torch.stack(state)
        action = torch.stack(action)
        next_state = torch.stack(next_state)
        reward = torch.stack(reward)
        terminated = torch.tensor(terminated).float().to(device)


        with torch.no_grad():
           target_q = reward + (1 - terminated) * self.gamma * target_dqn(next_state).max(dim = 1)[0]

        current_q = policy_dqn(state).gather(dim = 1, index = action.unsqueeze(dim = 1)).squeeze()

                #loss
        loss = self.loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Train or Test Model')
    parser.add_argument('hyperparameters', help = '')
    parser.add_argument('--train', help = 'Training mode', action = 'store_true')

    args = parser.parse_args()

    dql = Agent(parame_set = args.hyperparameters)

    if args.train:
        dql.run(is_training = True)
    else:
        dql.run(is_training = False, render = 'human')