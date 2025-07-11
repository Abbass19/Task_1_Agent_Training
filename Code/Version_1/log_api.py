import os
import json
import numpy as np
import json

LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "models", "log.json")


def initialize_log_if_missing():
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

    if not os.path.exists(LOG_FILE_PATH):
        log_data = {
            "best_hyperparameters": None,
            "hyperparameter_trials": [],
            "agent_performance": []
        }
        with open(LOG_FILE_PATH, "w") as f:
            json.dump(log_data, f, indent=2)
        print("Initialized log.json")
    else:
        print("log.json already exists.")


import json
import os

def _load_log():
    if not os.path.exists(LOG_FILE_PATH):
        return {}

    try:
        with open(LOG_FILE_PATH, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("⚠️ Warning: Log file is corrupted or incomplete. Returning empty log.")
        return {}




def convert_ndarrays(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays(i) for i in obj]
    else:
        return obj

def _save_log(data):
    safe_data = convert_ndarrays(data)
    with open(LOG_FILE_PATH, "w") as f:
        json.dump(safe_data, f, indent=2)



def save_hyperparameter_result(result_entry: dict):
    """
    result_entry = {
        "result": 1096100576.0,
        "params": { "lr": ..., "gamma": ..., ... }
    }
    """
    log = _load_log()

    # Add to trials
    log["hyperparameter_trials"].append(result_entry)

    # Update best if necessary
    best = log["best_hyperparameters"]
    if best is None or result_entry["result"] > best["result"]:
        log["best_hyperparameters"] = result_entry

    _save_log(log)


def save_agent_result(performance_entry: dict):
    """
    performance_entry = {
        "train_profit": ...,
        "val_profit": ...,
        "test_profit": ...,
        "hyperparameters": { ... }
    }
    """
    log = _load_log()

    if "agent_performance" not in log:
        log["agent_performance"] = []
    log["agent_performance"].append(performance_entry)
    _save_log(log)


def load_best_hyperparameter():
    log = _load_log()
    best = log.get("best_hyperparameters", None)
    return best["params"] if best else None
