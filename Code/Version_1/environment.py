import os

import gym
from gym import spaces
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


class Environment(gym.Env):
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.current_step = 0

        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(1,), dtype=np.float32
        )

        self.inventory = 0
        self.initial_cash = 10000
        self.cash = self.initial_cash

    def reset(self):
        self.current_step = 0
        self.inventory = 0
        self.cash = self.initial_cash
        return self._get_observation()

    def _get_observation(self):
        price = self.data.loc[self.current_step, 'MPN5P']
        return np.array([price], dtype=np.float32)

    def step(self, action):
        done = False
        price = self._get_observation()
        prev_total_assets = self.cash + self.inventory * price

        if action == 1:  # Buy
            if self.cash >= price:
                self.inventory += 1
                self.cash -= price
        elif action == 2:  # Sell
            if self.inventory > 0:
                self.inventory -= 1
                self.cash += price

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        price = self._get_observation()
        total_assets = self.cash + self.inventory * price
        profit_change = total_assets - prev_total_assets
        reward = profit_change
        obs = self._get_observation()

        info = {
            'step': self.current_step,
            'inventory': self.inventory,
            'cash': self.cash,
            'price': price,
            'profit_change': profit_change,
            'action_taken': action
        }

        return obs, reward, done, info

    def render(self, mode='human'):
        price = self._get_observation()
        total_assets = self.cash + self.inventory * price
        print(f"Step: {self.current_step}")
        print(f"Cash: {self.cash:.2f}")
        print(f"Inventory: {self.inventory}")
        print(f"Price: {price:.2f}")
        print(f"Total Assets: {total_assets:.2f}")

    @classmethod
    def with_splits_time_series(cls, data: pd.DataFrame, n_splits=5,Normalization = True ):

        data , old_mean , old_std= dataloader()


        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(tscv.split(data))
        if n_splits < 3:
            raise ValueError("n_splits must be at least 3 to get train/val/test splits.")

        train_val_idx, test_idx = splits[-1]
        train_idx, val_idx = splits[-2]

        train_data = data.iloc[train_idx].reset_index(drop=True)
        val_data = data.iloc[val_idx].reset_index(drop=True)
        test_data = data.iloc[test_idx].reset_index(drop=True)

        train_env = cls(train_data)
        val_env = cls(val_data)
        test_env = cls(test_data)

        return train_env, val_env, test_env


def dataloader():
    csv_path = os.path.join(os.path.dirname(__file__), "my_data.csv")
    sheet = pd.read_csv(csv_path)
    data = sheet["MPN5P"]
    data = np.log1p(data)
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / std
    return data, mean, std


def denormalize( data, old_mean, old_std):
    data = data*old_std + old_mean
    data = np.expm1(data)
    return data
