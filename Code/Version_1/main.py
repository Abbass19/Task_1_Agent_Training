from model_manager import ModelManager
from ppo_agent import PPOAgent
from log_api import initialize_log_if_missing, save_agent_result
from train import  train_agent,image_one, image_two


def main():
    model_manager = ModelManager(model_class=PPOAgent)
    model_manager.display_info()
    initialize_log_if_missing()
    # train_agent("30_episode_original", no_episodes=30)
    image_two("30_episode_early_stopping_l2_regularization", no_episodes=30)


if __name__ == "__main__":
    main()