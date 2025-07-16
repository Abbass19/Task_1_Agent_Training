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


#This function returns the list for the TPG
def image_three(report_number="", no_episodes=60):
    """
    Train a PPO agent with Optuna hyperparameter tuning.
    Also generates a detailed performance report.
    """

    # Load data and split environments
    training_env, validation_env, testing_env = Environment.with_splits_time_series(data)


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
            prediction_targets = []  # Initialize here at start of episode

            for step in range(rollout_len):
                action, log_prob, _, _ = agent.get_action(obs)
                next_obs, reward, done, info = training_env.step(action)  # Capture info dict

                with torch.no_grad():
                    _, value, _ = agent.model(torch.FloatTensor(obs).unsqueeze(0))

                observations.append(obs)
                actions.append(action)
                log_probs_old.append(log_prob.detach())
                rewards.append(reward)
                dones.append(done)
                values.append(value.item())

                # Collect true target from env info for prediction head training
                true_target = info.get("MPN5P", 0.0)  # default to 0.0 if None
                if true_target is None:
                    true_target = 0.0
                prediction_targets.append(true_target)

                total_return += reward
                obs = next_obs
                if done:
                    break

            with torch.no_grad():
                _, last_value, _ = agent.model(torch.FloatTensor(obs).unsqueeze(0))

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

            # Convert prediction_targets list to tensor here inside episode
            prediction_targets_tensor = torch.tensor(prediction_targets, dtype=torch.float32)

            # Pass prediction_targets_tensor to update method
            agent.update(
                obs_tensor,
                actions_tensor,
                log_probs_tensor,
                returns_tensor,
                advantages,
                true_predictions=prediction_targets_tensor,
                epochs=update_epochs
            )

            # Validation loop for early stopping
            val_obs = validation_env.reset()
            val_done = False
            val_profit = 0
            while not val_done:
                with torch.no_grad():
                    val_action, _, _, _ = agent.get_action(val_obs)
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

    agent = PPOAgent_edited(obs_dim=obs_dim, act_dim=act_dim,
                            gamma=best_params["gamma"],
                            clip_epsilon=best_params["clip_epsilon"],
                            lr=best_params["lr"])

    rollout_len = best_params["rollout_len"]
    update_epochs = best_params["update_epochs"]

    # Main training loop (after Optuna)
    for episode in range(no_episodes):
        obs = training_env.reset()
        done = False
        total_return = 0

        observations, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []
        prediction_targets = []  # Initialize here at start of episode

        for step in range(rollout_len):
            action, log_prob, _, _ = agent.get_action(obs)
            next_obs, reward, done, info = training_env.step(action)

            with torch.no_grad():
                _, value, _ = agent.model(torch.FloatTensor(obs).unsqueeze(0))

            observations.append(obs)
            actions.append(action)
            log_probs_old.append(log_prob.detach())
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())

            # Collect true target for prediction training
            true_target = info.get("MPN5P", 0.0)
            if true_target is None:
                true_target = 0.0
            prediction_targets.append(true_target)

            total_return += reward
            obs = next_obs

            if done:
                break

        with torch.no_grad():
            _, last_value, _ = agent.model(torch.FloatTensor(obs).unsqueeze(0))

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

        prediction_targets_tensor = torch.tensor(prediction_targets, dtype=torch.float32)

        agent.update(
            obs_tensor,
            actions_tensor,
            log_probs_tensor,
            returns_tensor,
            advantages,
            true_predictions=prediction_targets_tensor,
            epochs=update_epochs
        )


    def evaluate(env, label="Unknown"):
        print(f"\n📊 Evaluating agent on {label}...")
        obs = env.reset()
        done = False
        initial_cash = training_env.cash if hasattr(training_env, "cash") else 10000
        step_count = 0
        profit_changes = []
        actions_taken = []
        predictions = []
        targets = []
        dates = []
        final_info = {}

        while not done:
            with torch.no_grad():
                action, _, _, prediction = agent.get_action(obs)
            obs, reward, done, info = env.step(action)

            profit_change = info.get("profit_change", 0)
            date = info.get("date")  # <<< date must be provided by env
            target = info.get("MPN5P")  # <<< true target must be provided by env

            profit_changes.append(profit_change)
            actions_taken.append(action)
            predictions.append(prediction)
            targets.append(target)
            dates.append(date)

            final_info = info
            step_count += 1

        return {
            "label": label,
            "profits": profit_changes,
            "actions": actions_taken,
            "predicted_targets": predictions,
            "actual_targets": targets,
            "dates": dates,
            "steps": step_count,
            "final_inventory": final_info.get("inventory", 0),
            "final_cash": final_info.get("cash", 0),
            "initial_cash": initial_cash  # <<< ADDED THIS
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


    # Evaluate phases
    train_metrics = evaluate(training_env, label="Training")
    val_metrics = evaluate(validation_env, label="Validation")
    test_metrics = evaluate(testing_env, label="Testing")


    # actual_target_training = train_metrics["actual_targets"]
    # actual_target_validation = val_metrics["actual_targets"]
    # actual_target_testing = test_metrics["actual_targets"]

    # actual_target = [actual_target_training, actual_target_validation, actual_target_testing]

    predicted_target_training = train_metrics["predicted_targets"]
    predicted_target_validation = val_metrics["predicted_targets"]
    predicted_target_testing = test_metrics["predicted_targets"]

    predicted_target = predicted_target_training + predicted_target_validation + predicted_target_testing
    predicted_target_numpy = []

    for phase_list in predicted_target:
        # phase_list is like predicted_target_training: a list of tensors
        # convert each tensor to scalar float with .item()
        scalar_values = [t.item() for t in phase_list]
        # convert to numpy array
        np_array = np.array(scalar_values)
        predicted_target_numpy.append(np_array)

    predicted_target_numpy = np.array([x[0] for x in predicted_target_numpy])


    write_report(best_params, train_metrics, val_metrics, test_metrics)

    # Prepare TPG signature from test set
    # Read and convert the dates from the 'DCP' column in the Excel-style format
    test_dates = pd.to_datetime(data['DCP'], format="%m/%d/%Y")

    # Convert to strings in the format the function expects: 'YYYY-MM-DD'
    test_dates_str = [dt.strftime("%Y-%m-%d") for dt in test_dates]

    # Slice if needed (e.g., to match 3705 entries)
    test_dates_str = test_dates_str[:3705]


    # predicted_target = test_metrics["predicted_targets"]
    actual_target = data.loc[:, 'MPN5P'].to_numpy()
    target = "MPN5P"
    initial_cash = test_metrics["initial_cash"]
    calculated_yield = float((test_metrics["final_cash"] - initial_cash) / initial_cash)

    actual_target_aligned = actual_target[:3705]


    TPG_signature = [predicted_target_numpy, actual_target_aligned, test_dates_str, target, calculated_yield]

    print("Preparing to return TPG_signature and metrics")
    return TPG_signature, [train_metrics["profits"], val_metrics["profits"], test_metrics["profits"]]


