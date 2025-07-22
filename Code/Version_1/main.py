import os
import shutil

from model_manager import ModelManager
from ppo_agent import PPOAgent
from log_api import initialize_log_if_missing
from train import image_three
from Code.Version_1.targets_plot_generator import *


def save_plots_to_default_folder(predicted_targets, actual_targets, dates, target, calculated_yield, folder_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(base_dir, "My_Pictures", folder_name)
    os.makedirs(output_folder, exist_ok=True)
    plot_addresses = image_addresses(predicted_targets, actual_targets, dates, target, calculated_yield)

    # Save images to the designated folder
    saved_files = {}
    for name, temp_path in plot_addresses.items():
        new_path = os.path.join(output_folder, f"{name}.png")
        shutil.copy(temp_path, new_path)
        os.remove(temp_path)
        saved_files[name] = new_path

def sweep():
    hidden_size = [30, 45, 60]
    lambda_pred = [0.5, 1 , 1.5]
    drop_out = [0, 0.1, 0.3, 0.5]
    Normalization = [True, False]

    total_combinations = len(hidden_size) * len(lambda_pred) * len(drop_out) * len(Normalization)
    completed = 0  # Counter for completed runs

    for hidden_size_iterator in hidden_size:
        for lambda_pred_iterator in lambda_pred:
            for drop_out_iterator in drop_out:
                for Normalization_value in Normalization:
                    completed += 1  # Increment counter
                    percent_done = (completed / total_combinations) * 100

                    print(f"\n🧠 Progress: {completed}/{total_combinations} "
                          f"({percent_done:.2f}%) completed.")

                    model_manager = ModelManager(model_class=PPOAgent)
                    model_manager.display_info()
                    initialize_log_if_missing()
                    folder_name, TPG_signature, metrics = image_three("", no_episodes=15, hidden_size=hidden_size_iterator,
                                                                      lambda_pred=lambda_pred_iterator, drop_out=drop_out_iterator, Normalization= Normalization_value)
                    predicted_target, actual_target, test_dates, target, calculated_yield = TPG_signature
                    save_plots_to_default_folder(predicted_target, actual_target, test_dates, target, calculated_yield, folder_name=folder_name)



def main():
    # sweep()

    folder_name, TPG_signature, metrics = image_three("", no_episodes=15, hidden_size=30,
                                                      lambda_pred=4, drop_out=1,
                                                      Normalization=True)
    predicted_target, actual_target, test_dates, target, calculated_yield = TPG_signature
    save_plots_to_default_folder(predicted_target, actual_target, test_dates, target, calculated_yield,
                                 folder_name=folder_name)
    # model_manager = ModelManager(model_class=PPOAgent)
    # model_manager.display_info()
    # initialize_log_if_missing()
    # folder_name, TPG_signature , metrics = image_three("30_episode_early_stopping_l2_regularization", no_episodes=15)
    # predicted_target, actual_target, test_dates, target, calculated_yield = TPG_signature
    #
    #
    # save_plots_to_default_folder(predicted_target, actual_target, test_dates, target, calculated_yield)


if __name__ == "__main__":
    main()


