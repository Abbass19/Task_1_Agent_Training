import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        # Shared base
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(64, act_dim),
            nn.Softmax(dim=-1)
        )

        # Value head
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.policy(x), self.value(x)


class PPOAgent:
    def __init__(self, obs_dim, act_dim, gamma=0.99, clip_epsilon=0.2, lr=3e-4):
        self.model = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

    def get_action(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        probs, _ = self.model(obs)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def compute_returns(self, rewards, dones, last_value):
        returns = []
        R = last_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        returns = np.array(returns, dtype=np.float64)  # <-- convert list to np.array first
        return torch.tensor(returns)

    def update(self, observations, actions, log_probs_old, returns, advantages, epochs=10):
        for _ in range(epochs):
            probs, values = self.model(observations)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            # Ratio for clipped loss
            ratio = torch.exp(log_probs - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(values.view(-1), returns.view(-1))
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def save(self, path):
        torch.save(self.model.state_dict(), path)


