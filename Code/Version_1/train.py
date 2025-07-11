import pandas as pd

from log_api import *
from model_manager import *
import torch
import optuna
from environment import Environment
from ppo_agent import PPOAgent
from ppo_agent_edited import PPOAgent_edited

#Some Storage Data
csv_path = os.path.join(os.path.dirname(__file__), "models", "my_data.csv")
data = pd.read_csv(csv_path)

obs_dim = 1
act_dim = 3



#Train with Optuna and txt Reporting (Works for 30 Plus episodes)
def train_agent(report_number="", no_episodes=60):
    """
    Train a PPO agent with Optuna hyperparameter tuning.
    Also generates a detailed performance report.
    """

    # Load data and split environments
    training_env, validation_env, testing_env = Environment.with_splits_time_series(data)
    print(f"[INFO] Training env length: {len(training_env.data)}")
    print(f"[INFO] Validation env length: {len(validation_env.data)}")
    print(f"[INFO] Testing env length: {len(testing_env.data)}")

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.90, 0.999)
        clip_epsilon = trial.suggest_float("clip_epsilon", 0.1, 0.3)
        rollout_len = trial.suggest_categorical("rollout_len", [128, 256, 512])
        update_epochs = trial.suggest_int("update_epochs", 5, 15)

        agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, gamma=gamma, clip_epsilon=clip_epsilon, lr=lr)

        for episode in range(no_episodes):
            obs = training_env.reset()
            done = False
            total_return = 0

            observations, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []

            for step in range(rollout_len):
                action, log_prob, _ = agent.get_action(obs)
                next_obs, reward, done, _ = training_env.step(action)

                with torch.no_grad():
                    _, value = agent.model(torch.FloatTensor(obs).unsqueeze(0))

                observations.append(obs)
                actions.append(action)
                log_probs_old.append(log_prob.detach())
                rewards.append(reward)
                dones.append(done)
                values.append(value.item())

                total_return += reward
                obs = next_obs

                if done:
                    break

            with torch.no_grad():
                _, last_value = agent.model(torch.FloatTensor(obs).unsqueeze(0))
            returns = agent.compute_returns(rewards, dones, last_value.item())

            obs_tensor = torch.FloatTensor(np.array(observations))
            actions_tensor = torch.tensor(actions)
            log_probs_tensor = torch.stack(log_probs_old).float()
            if isinstance(returns, torch.Tensor):
                returns_tensor = returns.detach().clone().float()
            else:
                returns_tensor = torch.tensor(returns, dtype=torch.float32)

            values_tensor = torch.tensor(values, dtype=torch.float32)

            advantages = returns_tensor - values_tensor
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            agent.update(obs_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantages, epochs=update_epochs)

        obs = validation_env.reset()
        done = False
        val_profit = 0
        while not done:
            with torch.no_grad():
                action, _, _ = agent.get_action(obs)
            obs, reward, done, _ = validation_env.step(action)
            val_profit += reward

        return val_profit

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)

    best_params = study.best_trial.params
    print(f"[INFO] Best hyperparameters from Optuna: {best_params}")

    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim,
                     gamma=best_params["gamma"],
                     clip_epsilon=best_params["clip_epsilon"],
                     lr=best_params["lr"])

    rollout_len = best_params["rollout_len"]
    update_epochs = best_params["update_epochs"]

    for episode in range(no_episodes):
        obs = training_env.reset()
        done = False
        total_return = 0

        observations, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []

        for step in range(rollout_len):
            action, log_prob, _ = agent.get_action(obs)
            next_obs, reward, done, _ = training_env.step(action)

            with torch.no_grad():
                _, value = agent.model(torch.FloatTensor(obs).unsqueeze(0))

            observations.append(obs)
            actions.append(action)
            log_probs_old.append(log_prob.detach())
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())

            total_return += reward
            obs = next_obs

            if done:
                break

        with torch.no_grad():
            _, last_value = agent.model(torch.FloatTensor(obs).unsqueeze(0))
        returns = agent.compute_returns(rewards, dones, last_value.item())

        obs_tensor = torch.FloatTensor(np.array(observations))
        actions_tensor = torch.tensor(actions)
        log_probs_tensor = torch.stack(log_probs_old).float()
        if isinstance(returns, torch.Tensor):
            returns_tensor = returns.detach().clone().float()
        else:
            returns_tensor = torch.tensor(returns, dtype=torch.float32)

        values_tensor = torch.tensor(values, dtype=torch.float32)

        advantages = returns_tensor - values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        agent.update(obs_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantages, epochs=update_epochs)
        print(f"Train_Two_Agent Speaking : Episode {episode + 1} finished. Total return: {total_return}")

    def evaluate(env, label="Unknown"):
        print(f"\n📊 Evaluating agent on {label}...")
        obs = env.reset()
        done = False
        step_count = 0
        profit_changes = []
        actions_taken = []
        final_info = {}

        while not done:
            with torch.no_grad():
                action, _, _ = agent.get_action(obs)
            obs, reward, done, info = env.step(action)

            profit_change = info.get("profit_change", 0)
            profit_changes.append(profit_change)
            actions_taken.append(action)
            final_info = info
            step_count += 1

        return {
            "label": label,
            "profits": profit_changes,
            "actions": actions_taken,
            "steps": step_count,
            "final_inventory": final_info.get("inventory", 0),
            "final_cash": final_info.get("cash", 0)
        }

    def summarize_performance(metrics):
        profits = np.array(metrics["profits"])
        actions = np.array(metrics["actions"])

        total_profit = np.sum(profits)
        avg_profit = np.mean(profits)
        max_gain = np.max(profits)
        max_loss = np.min(profits)
        pos_steps = np.sum(profits > 0)
        neg_steps = np.sum(profits < 0)
        neut_steps = np.sum(profits == 0)
        steps = metrics["steps"]

        buy = np.sum(actions == 1)
        sell = np.sum(actions == 2)
        hold = np.sum(actions == 0)

        buy_pct = 100 * buy / steps
        sell_pct = 100 * sell / steps
        hold_pct = 100 * hold / steps

        return f"""
--- {metrics['label']} Phase ---
Total Profit: [{total_profit}]
Avg Profit/Step: [{avg_profit}]
Max Gain: [{max_gain}]
Max Loss: [{max_loss}]
Steps: {steps}
Positive Steps: [{pos_steps}]
Negative Steps: [{neg_steps}]
Neutral Steps: [{neut_steps}]
Buy: {buy} times ({buy_pct}%)
Sell: {sell} times ({sell_pct}%)
Hold: {hold} times ({hold_pct}%)
Final Inventory: {metrics["final_inventory"]}
Final Cash: [{metrics["final_cash"]}]
""".strip()

    def write_report(best_params, train_metrics, val_metrics, test_metrics):
        report_dir = r"C:\Users\Abbass Zahreddine\Documents\GitHub\Task_1_Agent_Training\Code\Version_1\agent_reports"
        os.makedirs(report_dir, exist_ok=True)
        filename = f"agent_report_{report_number}.txt"
        path = os.path.join(report_dir, filename)

        report = f"""==================== AGENT PERFORMANCE REPORT ====================
Used Hyperparameters: {best_params}

{summarize_performance(train_metrics)}

{summarize_performance(val_metrics)}

{summarize_performance(test_metrics)}
"""
        with open(path, "w") as f:
            f.write(report)
        print(f"\n📄 Report saved to: {path}")

    train_metrics = evaluate(training_env, label="Training")
    val_metrics = evaluate(validation_env, label="Validation")
    test_metrics = evaluate(testing_env, label="Testing")

    write_report(best_params, train_metrics, val_metrics, test_metrics)

    return [train_metrics["profits"], val_metrics["profits"], test_metrics["profits"]]

def image_one(report_number="", no_episodes=60):
    """
    Train a PPO agent with Optuna hyperparameter tuning.
    Also generates a detailed performance report.
    """

    # Load data and split environments
    training_env, validation_env, testing_env = Environment.with_splits_time_series(data)
    print(f"[INFO] Training env length: {len(training_env.data)}")
    print(f"[INFO] Validation env length: {len(validation_env.data)}")
    print(f"[INFO] Testing env length: {len(testing_env.data)}")

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.90, 0.999)
        clip_epsilon = trial.suggest_float("clip_epsilon", 0.1, 0.3)
        rollout_len = trial.suggest_categorical("rollout_len", [128, 256, 512])
        update_epochs = trial.suggest_int("update_epochs", 5, 15)

        agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, gamma=gamma, clip_epsilon=clip_epsilon, lr=lr)

        patience = 5
        best_val_profit = float("-inf")
        wait = 0

        for episode in range(no_episodes):
            obs = training_env.reset()
            done = False
            total_return = 0

            observations, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []

            for step in range(rollout_len):
                action, log_prob, _ = agent.get_action(obs)
                next_obs, reward, done, _ = training_env.step(action)

                with torch.no_grad():
                    _, value = agent.model(torch.FloatTensor(obs).unsqueeze(0))

                observations.append(obs)
                actions.append(action)
                log_probs_old.append(log_prob.detach())
                rewards.append(reward)
                dones.append(done)
                values.append(value.item())

                total_return += reward
                obs = next_obs
                if done:
                    break

            with torch.no_grad():
                _, last_value = agent.model(torch.FloatTensor(obs).unsqueeze(0))
            returns = agent.compute_returns(rewards, dones, last_value.item())

            obs_tensor = torch.FloatTensor(np.array(observations))
            actions_tensor = torch.tensor(actions)
            log_probs_tensor = torch.stack(log_probs_old).float()
            returns_tensor = returns.detach().clone().float() if isinstance(returns, torch.Tensor) else torch.tensor(
                returns, dtype=torch.float32)
            values_tensor = torch.tensor(values, dtype=torch.float32)

            advantages = returns_tensor - values_tensor
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            agent.update(obs_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantages, epochs=update_epochs)

            # 🔍 Evaluate on validation
            val_obs = validation_env.reset()
            val_done = False
            val_profit = 0
            while not val_done:
                with torch.no_grad():
                    val_action, _, _ = agent.get_action(val_obs)
                val_obs, val_reward, val_done, _ = validation_env.step(val_action)
                val_profit += val_reward

            if val_profit > best_val_profit:
                best_val_profit = val_profit
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"[EARLY STOPPING] Episode {episode + 1}: No improvement in {patience} episodes.")
                    break

        return best_val_profit

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)

    best_params = study.best_trial.params
    print(f"[INFO] Best hyperparameters from Optuna: {best_params}")

    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim,
                     gamma=best_params["gamma"],
                     clip_epsilon=best_params["clip_epsilon"],
                     lr=best_params["lr"])

    rollout_len = best_params["rollout_len"]
    update_epochs = best_params["update_epochs"]

    for episode in range(no_episodes):
        obs = training_env.reset()
        done = False
        total_return = 0

        observations, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []

        for step in range(rollout_len):
            action, log_prob, _ = agent.get_action(obs)
            next_obs, reward, done, _ = training_env.step(action)

            with torch.no_grad():
                _, value = agent.model(torch.FloatTensor(obs).unsqueeze(0))

            observations.append(obs)
            actions.append(action)
            log_probs_old.append(log_prob.detach())
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())

            total_return += reward
            obs = next_obs

            if done:
                break

        with torch.no_grad():
            _, last_value = agent.model(torch.FloatTensor(obs).unsqueeze(0))
        returns = agent.compute_returns(rewards, dones, last_value.item())

        obs_tensor = torch.FloatTensor(np.array(observations))
        actions_tensor = torch.tensor(actions)
        log_probs_tensor = torch.stack(log_probs_old).float()
        if isinstance(returns, torch.Tensor):
            returns_tensor = returns.detach().clone().float()
        else:
            returns_tensor = torch.tensor(returns, dtype=torch.float32)

        values_tensor = torch.tensor(values, dtype=torch.float32)

        advantages = returns_tensor - values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        agent.update(obs_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantages, epochs=update_epochs)
        print(f"Train_Two_Agent Speaking : Episode {episode + 1} finished. Total return: {total_return}")

    def evaluate(env, label="Unknown"):
        print(f"\n📊 Evaluating agent on {label}...")
        obs = env.reset()
        done = False
        step_count = 0
        profit_changes = []
        actions_taken = []
        final_info = {}

        while not done:
            with torch.no_grad():
                action, _, _ = agent.get_action(obs)
            obs, reward, done, info = env.step(action)

            profit_change = info.get("profit_change", 0)
            profit_changes.append(profit_change)
            actions_taken.append(action)
            final_info = info
            step_count += 1

        return {
            "label": label,
            "profits": profit_changes,
            "actions": actions_taken,
            "steps": step_count,
            "final_inventory": final_info.get("inventory", 0),
            "final_cash": final_info.get("cash", 0)
        }

    def summarize_performance(metrics):
        profits = np.array(metrics["profits"])
        actions = np.array(metrics["actions"])

        total_profit = np.sum(profits)
        avg_profit = np.mean(profits)
        max_gain = np.max(profits)
        max_loss = np.min(profits)
        pos_steps = np.sum(profits > 0)
        neg_steps = np.sum(profits < 0)
        neut_steps = np.sum(profits == 0)
        steps = metrics["steps"]

        buy = np.sum(actions == 1)
        sell = np.sum(actions == 2)
        hold = np.sum(actions == 0)

        buy_pct = 100 * buy / steps
        sell_pct = 100 * sell / steps
        hold_pct = 100 * hold / steps

        return f"""
--- {metrics['label']} Phase ---
Total Profit: [{total_profit}]
Avg Profit/Step: [{avg_profit}]
Max Gain: [{max_gain}]
Max Loss: [{max_loss}]
Steps: {steps}
Positive Steps: [{pos_steps}]
Negative Steps: [{neg_steps}]
Neutral Steps: [{neut_steps}]
Buy: {buy} times ({buy_pct}%)
Sell: {sell} times ({sell_pct}%)
Hold: {hold} times ({hold_pct}%)
Final Inventory: {metrics["final_inventory"]}
Final Cash: [{metrics["final_cash"]}]
""".strip()

    def write_report(best_params, train_metrics, val_metrics, test_metrics):
        report_dir = r"C:\Users\Abbass Zahreddine\Documents\GitHub\Task_1_Agent_Training\Code\Version_1\agent_reports"
        os.makedirs(report_dir, exist_ok=True)
        filename = f"agent_report_{report_number}.txt"
        path = os.path.join(report_dir, filename)

        report = f"""==================== AGENT PERFORMANCE REPORT ====================
Used Hyperparameters: {best_params}

{summarize_performance(train_metrics)}

{summarize_performance(val_metrics)}

{summarize_performance(test_metrics)}
"""
        with open(path, "w") as f:
            f.write(report)
        print(f"\n📄 Report saved to: {path}")

    train_metrics = evaluate(training_env, label="Training")
    val_metrics = evaluate(validation_env, label="Validation")
    test_metrics = evaluate(testing_env, label="Testing")

    write_report(best_params, train_metrics, val_metrics, test_metrics)

    return [train_metrics["profits"], val_metrics["profits"], test_metrics["profits"]]

def image_two(report_number="", no_episodes=60):
    """
    Train a PPO agent with Optuna hyperparameter tuning.
    Also generates a detailed performance report.
    """

    # Load data and split environments
    training_env, validation_env, testing_env = Environment.with_splits_time_series(data)
    print(f"[INFO] Training env length: {len(training_env.data)}")
    print(f"[INFO] Validation env length: {len(validation_env.data)}")
    print(f"[INFO] Testing env length: {len(testing_env.data)}")

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.90, 0.999)
        clip_epsilon = trial.suggest_float("clip_epsilon", 0.1, 0.3)
        rollout_len = trial.suggest_categorical("rollout_len", [128, 256, 512])
        update_epochs = trial.suggest_int("update_epochs", 5, 15)

        agent = PPOAgent_edited(obs_dim=obs_dim, act_dim=act_dim, gamma=gamma, clip_epsilon=clip_epsilon, lr=lr)

        patience = 5
        best_val_profit = float("-inf")
        wait = 0

        for episode in range(no_episodes):
            obs = training_env.reset()
            done = False
            total_return = 0

            observations, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []

            for step in range(rollout_len):
                action, log_prob, _ = agent.get_action(obs)
                next_obs, reward, done, _ = training_env.step(action)

                with torch.no_grad():
                    _, value = agent.model(torch.FloatTensor(obs).unsqueeze(0))

                observations.append(obs)
                actions.append(action)
                log_probs_old.append(log_prob.detach())
                rewards.append(reward)
                dones.append(done)
                values.append(value.item())

                total_return += reward
                obs = next_obs
                if done:
                    break

            with torch.no_grad():
                _, last_value = agent.model(torch.FloatTensor(obs).unsqueeze(0))
            returns = agent.compute_returns(rewards, dones, last_value.item())

            obs_tensor = torch.FloatTensor(np.array(observations))
            actions_tensor = torch.tensor(actions)
            log_probs_tensor = torch.stack(log_probs_old).float()
            returns_tensor = returns.detach().clone().float() if isinstance(returns, torch.Tensor) else torch.tensor(
                returns, dtype=torch.float32)
            values_tensor = torch.tensor(values, dtype=torch.float32)

            advantages = returns_tensor - values_tensor
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            agent.update(obs_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantages, epochs=update_epochs)

            # 🔍 Evaluate on validation
            val_obs = validation_env.reset()
            val_done = False
            val_profit = 0
            while not val_done:
                with torch.no_grad():
                    val_action, _, _ = agent.get_action(val_obs)
                val_obs, val_reward, val_done, _ = validation_env.step(val_action)
                val_profit += val_reward

            if val_profit > best_val_profit:
                best_val_profit = val_profit
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"[EARLY STOPPING] Episode {episode + 1}: No improvement in {patience} episodes.")
                    break

        return best_val_profit

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)

    best_params = study.best_trial.params
    print(f"[INFO] Best hyperparameters from Optuna: {best_params}")

    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim,
                     gamma=best_params["gamma"],
                     clip_epsilon=best_params["clip_epsilon"],
                     lr=best_params["lr"])

    rollout_len = best_params["rollout_len"]
    update_epochs = best_params["update_epochs"]

    for episode in range(no_episodes):
        obs = training_env.reset()
        done = False
        total_return = 0

        observations, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []

        for step in range(rollout_len):
            action, log_prob, _ = agent.get_action(obs)
            next_obs, reward, done, _ = training_env.step(action)

            with torch.no_grad():
                _, value = agent.model(torch.FloatTensor(obs).unsqueeze(0))

            observations.append(obs)
            actions.append(action)
            log_probs_old.append(log_prob.detach())
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())

            total_return += reward
            obs = next_obs

            if done:
                break

        with torch.no_grad():
            _, last_value = agent.model(torch.FloatTensor(obs).unsqueeze(0))
        returns = agent.compute_returns(rewards, dones, last_value.item())

        obs_tensor = torch.FloatTensor(np.array(observations))
        actions_tensor = torch.tensor(actions)
        log_probs_tensor = torch.stack(log_probs_old).float()
        if isinstance(returns, torch.Tensor):
            returns_tensor = returns.detach().clone().float()
        else:
            returns_tensor = torch.tensor(returns, dtype=torch.float32)

        values_tensor = torch.tensor(values, dtype=torch.float32)

        advantages = returns_tensor - values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        agent.update(obs_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantages, epochs=update_epochs)
        print(f"Train_Two_Agent Speaking : Episode {episode + 1} finished. Total return: {total_return}")

    def evaluate(env, label="Unknown"):
        print(f"\n📊 Evaluating agent on {label}...")
        obs = env.reset()
        done = False
        step_count = 0
        profit_changes = []
        actions_taken = []
        final_info = {}

        while not done:
            with torch.no_grad():
                action, _, _ = agent.get_action(obs)
            obs, reward, done, info = env.step(action)

            profit_change = info.get("profit_change", 0)
            profit_changes.append(profit_change)
            actions_taken.append(action)
            final_info = info
            step_count += 1

        return {
            "label": label,
            "profits": profit_changes,
            "actions": actions_taken,
            "steps": step_count,
            "final_inventory": final_info.get("inventory", 0),
            "final_cash": final_info.get("cash", 0)
        }

    def summarize_performance(metrics):
        profits = np.array(metrics["profits"])
        actions = np.array(metrics["actions"])

        total_profit = np.sum(profits)
        avg_profit = np.mean(profits)
        max_gain = np.max(profits)
        max_loss = np.min(profits)
        pos_steps = np.sum(profits > 0)
        neg_steps = np.sum(profits < 0)
        neut_steps = np.sum(profits == 0)
        steps = metrics["steps"]

        buy = np.sum(actions == 1)
        sell = np.sum(actions == 2)
        hold = np.sum(actions == 0)

        buy_pct = 100 * buy / steps
        sell_pct = 100 * sell / steps
        hold_pct = 100 * hold / steps

        return f"""
--- {metrics['label']} Phase ---
Total Profit: [{total_profit}]
Avg Profit/Step: [{avg_profit}]
Max Gain: [{max_gain}]
Max Loss: [{max_loss}]
Steps: {steps}
Positive Steps: [{pos_steps}]
Negative Steps: [{neg_steps}]
Neutral Steps: [{neut_steps}]
Buy: {buy} times ({buy_pct}%)
Sell: {sell} times ({sell_pct}%)
Hold: {hold} times ({hold_pct}%)
Final Inventory: {metrics["final_inventory"]}
Final Cash: [{metrics["final_cash"]}]
""".strip()

    def write_report(best_params, train_metrics, val_metrics, test_metrics):
        report_dir = r"C:\Users\Abbass Zahreddine\Documents\GitHub\Task_1_Agent_Training\Code\Version_1\agent_reports"
        os.makedirs(report_dir, exist_ok=True)
        filename = f"agent_report_{report_number}.txt"
        path = os.path.join(report_dir, filename)

        report = f"""==================== AGENT PERFORMANCE REPORT ====================
Used Hyperparameters: {best_params}

{summarize_performance(train_metrics)}

{summarize_performance(val_metrics)}

{summarize_performance(test_metrics)}
"""
        with open(path, "w") as f:
            f.write(report)
        print(f"\n📄 Report saved to: {path}")

    train_metrics = evaluate(training_env, label="Training")
    val_metrics = evaluate(validation_env, label="Validation")
    test_metrics = evaluate(testing_env, label="Testing")

    write_report(best_params, train_metrics, val_metrics, test_metrics)

    return [train_metrics["profits"], val_metrics["profits"], test_metrics["profits"]]



