import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis
from typing import Union, List

from Code.Version_3.targets_plot_generator import save_plots_to_default_folder


def dataloader(csv_path = None, target_column: str = "MPN5P"):
    csv_path = os.path.join(os.path.dirname(__file__), "my_data.csv")
    sheet = pd.read_csv(csv_path)
    data = sheet[target_column]
    data = np.log1p(data)
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / std
    return data, mean, std

def denormalize( data, old_mean, old_std):
    data = data*old_std + old_mean
    data = np.expm1(data)
    return data


def analyze_distribution(data):
    print("📊 Data Distribution Analysis\n")

    # Basic Stats
    mean = np.mean(data)
    std = np.std(data)
    cv = std / mean if mean != 0 else float('inf')
    data_skew = skew(data)
    data_kurtosis = kurtosis(data)

    # IQR and Outliers
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data < lower_bound) | (data > upper_bound)
    outlier_ratio = np.sum(outliers) / len(data)

    # Percentile Ratios
    p70 = np.percentile(data, 70)
    p99 = np.percentile(data, 99)
    p_ratio = p99 / p70 if p70 != 0 else float('inf')

    # Judgments
    judgments = {
        'CV':          ("✅ Good" if cv < 1.0 else "⚠️ High variability"),
        'Skewness':    ("✅ Symmetric" if abs(data_skew) < 0.5 else "⚠️ Skewed"),
        'Kurtosis':    ("✅ Normal-like" if -1 < data_kurtosis < 3 else "⚠️ Extreme tails"),
        'Outliers':    ("✅ Low outliers" if outlier_ratio < 0.05 else "⚠️ Too many outliers"),
        'P-Ratio':     ("✅ Stable spread" if p_ratio < 2.0 else "⚠️ Top-end dominance")
    }

    # Print Metrics & Judgments
    print(f"🔸 Mean: {mean:.4f}")
    print(f"🔸 Std Dev: {std:.4f}")
    print(f"🔸 Coefficient of Variation (CV): {cv:.4f} → {judgments['CV']}")
    print(f"🔸 Skewness: {data_skew:.4f} → {judgments['Skewness']}")
    print(f"🔸 Kurtosis: {data_kurtosis:.4f} → {judgments['Kurtosis']}")
    print(f"🔸 IQR: {IQR:.4f}")
    print(f"🔸 Outlier Ratio (Tukey's rule): {outlier_ratio * 100:.2f}% → {judgments['Outliers']}")
    print(f"🔸 99th / 70th Percentile: {p_ratio:.2f} → {judgments['P-Ratio']}")
    print()

    # Overall Readiness
    readiness = all("✅" in v for v in judgments.values())
    if readiness:
        print("✅ The data appears ready for model training.\n")
    else:
        print("⚠️ Data may require preprocessing (transformation or cleaning) before training.\n")

    # Plots
    plt.figure(figsize=(12, 5))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(data, bins=40, edgecolor='black', alpha=0.7)
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    # CDF
    plt.subplot(1, 2, 2)
    sorted_data = np.sort(data)
    cdf = np.arange(len(data)) / len(data)
    plt.plot(sorted_data, cdf)
    plt.title('CDF (Cumulative Distribution Function)')
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def smooth_data(data, window_size=9):

    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 1")

    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(data, kernel, mode='same')
    return smoothed

def rolling_forecast_origin_split(data, n_splits=5, val_size=20):

    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        data = data.to_numpy()

    total_len = len(data)
    train_val_pairs = []

    for i in range(n_splits):
        train_end = total_len - val_size * (n_splits - i)
        val_start = train_end
        val_end = val_start + val_size

        train = data[:train_end]
        val = data[val_start:val_end]

        train_val_pairs.append((train, val))

    return train_val_pairs


def get_dates_from_csv(csv_path=None, date_column: str = "DCP"):
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "my_data.csv")
    sheet = pd.read_csv(csv_path)
    date_series = pd.to_datetime(sheet[date_column], format="%m/%d/%Y")
    date_str_list = [dt.strftime("%Y-%m-%d") for dt in date_series]
    return date_str_list

def flatten_predictions(preds):
    # Handles torch, numpy, or nested lists
    flat = []
    try:
        for p in preds:
            if hasattr(p, 'item'):
                flat.append(p.item())
            elif isinstance(p, (list, tuple, np.ndarray)):
                flat.append(p[0] if hasattr(p[0], 'item') else p[0])
            else:
                flat.append(p)
    except Exception as e:
        raise ValueError(f"Could not flatten predictions: {e}")
    return np.array(flat)

def to_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    if hasattr(arr, 'numpy'):
        return arr.numpy()
    if isinstance(arr, (list, tuple)):
        return np.array(arr)
    raise ValueError("Unsupported data type for actual values.")

def run_TPG_link(predictions: Union[list, np.ndarray],actual: Union[list, np.ndarray],
    target: str = "TARGET",csv_path: str =  os.path.join(os.path.dirname(__file__), "my_data.csv"),folder_name: str = "TPG_Output"):
    # Step 1: Parse predictions
    predicted_array = flatten_predictions(predictions)

    # Step 2: Parse actual values
    actual_array = to_numpy(actual)

    # Step 3: Load and format date list
    date_list = get_dates_from_csv(csv_path)

    # Step 4: Align lengths
    min_len = min(len(predicted_array), len(actual_array), len(date_list))
    predicted_array = predicted_array[:min_len]
    actual_array = actual_array[:min_len]
    date_list = date_list[:min_len]

    # Step 5: Dummy yield (can update later)
    calculated_yield = 0.0

    # Step 6: Save plots using internal utility
    save_plots_to_default_folder(
        predicted_array,
        actual_array,
        date_list,
        target,
        calculated_yield,
        folder_name
    )
