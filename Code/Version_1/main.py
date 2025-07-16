import os
import shutil

from model_manager import ModelManager
from ppo_agent import PPOAgent
from log_api import initialize_log_if_missing, save_agent_result
from train import train_agent, image_one, image_two, image_three
from TPG.targets_plot_generator import *


def save_plots_to_default_folder(predicted_targets, actual_targets, dates, target, calculated_yield):
    # Folder named "My_Pictures" next to the running Python script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(base_dir, "My_Pictures")

    # Create the folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get temp plot paths
    plot_addresses = image_addresses(predicted_targets, actual_targets, dates, target, calculated_yield)

    saved_files = {}
    for name, temp_path in plot_addresses.items():
        # Define permanent file path
        new_path = os.path.join(output_folder, f"{name}.png")

        # Copy temp file to permanent location
        shutil.copy(temp_path, new_path)

        # Remove temp file after copying
        os.remove(temp_path)

        saved_files[name] = new_path

    return saved_files


def main():
    model_manager = ModelManager(model_class=PPOAgent)
    model_manager.display_info()
    initialize_log_if_missing()
    TPG_signature , metrics = image_three("30_episode_early_stopping_l2_regularization", no_episodes=2)
    predicted_target, actual_target, test_dates, target, calculated_yield = TPG_signature



    save_plots_to_default_folder(predicted_target, actual_target, test_dates, target, calculated_yield)


if __name__ == "__main__":
    main()