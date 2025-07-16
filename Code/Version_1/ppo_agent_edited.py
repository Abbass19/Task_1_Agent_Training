import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()

        # Shared feature extractor
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 45),
            nn.ReLU(),
            nn.Linear(45, 45),
            nn.ReLU(),
        )

        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(45, act_dim),
            nn.Softmax(dim=-1),
        )

        # Value head
        self.value = nn.Linear(45, 1)

        self.prediction = nn.Linear(45, 1)

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        return self.policy(x), self.value(x), self.prediction(x)


class PPOAgent_edited:
    """Proximal Policy Optimization (PPO) agent with optional L2 regularisation."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,  # L2 regularisation coefficient
        entropy_coef: float = 0.01,  # entropy regularisation coefficient
    ) -> None:
        self.model = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef

    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------
    def get_action(self, obs):
        """Select an action using the current policy."""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        probs, value , prediction = self.model(obs_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy(), prediction

    # ------------------------------------------------------------------
    # Return & advantage utilities
    # ------------------------------------------------------------------
    def compute_returns(self, rewards, dones, last_value):
        """Compute discounted returns with bootstrapping."""
        returns = []
        R = last_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        returns = np.array(returns, dtype=np.float32)
        return torch.from_numpy(returns)

    # ------------------------------------------------------------------
    # Optimisation step
    # ------------------------------------------------------------------
    def update(self, observations, actions, log_probs_old, returns, advantages, *, epochs: int = 10, true_predictions, lambda_pred=0.5):
        for _ in range(epochs):
            probs, values, preds = self.model(observations)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            # PPO clipped surrogate objective
            ratio = torch.exp(log_probs - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss
            value_loss = nn.functional.mse_loss(values.view(-1), returns.view(-1))

            prediction_loss = nn.functional.mse_loss(preds.view(-1), true_predictions.view(-1))

            loss = policy_loss + 0.5 * value_loss + lambda_pred * prediction_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save network weights to disk."""
        torch.save(self.model.state_dict(), path)
