import os
import json
import torch
from typing import Any, Dict
import torch.nn as nn


class StoredModel:
    """
    A wrapper that holds a model, its score (typically val_profit),
    and related training info.
    """
    def __init__(self, model: nn.Module, result_info: Dict[str, Any]):
        self.model = model
        self.result_info = result_info
        self.score = result_info.get("val_profit", 0.0)

    def save_to_disk(self, path: str):
        """
        Saves the model and its info to the given folder path.
        """
        model_path = os.path.join(path, "model.pth")
        info_path = os.path.join(path, "info.json")
        torch.save(self.model.state_dict(), model_path)
        with open(info_path, "w") as f:
            json.dump(self.result_info, f, indent=2)

    @staticmethod
    def load_from_disk(model_class: Any, path: str):
        """
        Loads model and info from disk and returns a StoredModel instance.
        """
        model = model_class()
        model_path = os.path.join(path, "model.pth")
        info_path = os.path.join(path, "info.json")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        with open(info_path, "r") as f:
            result_info = json.load(f)
        return StoredModel(model, result_info)


class ModelManager:
    """
    Manages storage of up to 5 best models sorted by val_profit.
    """

    def __init__(self, model_class: Any):
        self.model_class = model_class
        self.models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def _get_existing_models(self):
        """
        Loads all models from disk and returns list of (index, StoredModel)
        sorted by score descending.
        """
        models = []
        for i in range(1, 6):
            folder = os.path.join(self.models_dir, f"model_{i}")
            if os.path.exists(folder):
                try:
                    sm = StoredModel.load_from_disk(self.model_class, folder)
                    models.append((i, sm))
                except Exception as e:
                    print(f"Failed to load model_{i}: {e}")
        # Sort by score descending
        models.sort(key=lambda x: x[1].score, reverse=True)
        return models

    def save_model(self, model: nn.Module, result_info: Dict[str, Any]) -> None:
        """
        Saves the model if it's among the top 5 best (by val_profit).
        Shuffles others down if necessary.
        """
        pass  # To be implemented step-by-step

    def load_model(self, index: int) -> nn.Module:
        """
        Loads the model ranked at `index` (1=best, 5=worst).
        Returns the model object.
        """
        folder = os.path.join(self.models_dir, f"model_{index}")
        if not os.path.exists(folder):
            raise ValueError(f"No model saved at rank {index}.")
        stored = StoredModel.load_from_disk(self.model_class, folder)
        return stored.model

    def display_info(self) -> None:
        """
        Prints out info (val_profit, train_profit, test_profit) for each saved model.
        """
        models = self._get_existing_models()
        if not models:
            print("No models saved yet.")
            return

        print("=== Stored Models ===")
        for rank, stored in models:
            info = stored.result_info
            print(f"Rank {rank}: val={info.get('val_profit')}, train={info.get('train_profit')}, test={info.get('test_profit')}")
