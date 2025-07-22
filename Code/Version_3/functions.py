import os
import random
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.regression import MeanAbsolutePercentageError

from Code.Version_3.preprocessing_functions import dataloader, rolling_forecast_origin_split, denormalize, \
    get_dates_from_csv, run_TPG_link
from defined_modules import SimpleBaseLine


#Fixed Seed :
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)




def split_data(data, train_ratio: float = 0.6):
    length = len(data)
    training_number = int(np.floor(train_ratio * length))
    training_data = data[0:training_number]         # up to but not including training_number
    testing_data = data[training_number:]           # from training_number to end
    return training_data, testing_data


def train_loop(model: nn.Module, training_data, epochs: int=5,batch_size: int = 32, lr: float = 0.0001, l2_lambda: float = 0.01, window_size: int = 8, patience: int = 5, alpha = 0.5):


    #Recording variables
    per_epoch_loss = []
    all_time_loss =[]
    prediction = None
    patience_counter = 0


    #Loss Function criteria
    criterion = nn.MSELoss()
    mape = MeanAbsolutePercentageError()


    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    best_val_loss = np.finfo(np.float32).max

    for epoch in range(epochs):

        if patience_counter==patience:
            print(f"Early stopping rerun at epoch {epoch+1}")
            return model, all_time_loss, per_epoch_loss, prediction
        prediction = None
        print(f"Training model in epoch {epoch+1} out of {epochs}")
        epoch_loss = 0
        batch_window_tuple_list = batch_window_generator(training_data, batch_size=batch_size, window_size=window_size)
        for data_tuple in batch_window_tuple_list:
            batch, window = data_tuple
            optimizer.zero_grad()
            batch_prediction = []

            if len(batch) == 0:
                continue

            for i in range(len(batch)):
                input_window = move_on(batch, window, i)
                input_window = input_window.unsqueeze(0)
                pred = model(input_window)
                batch_prediction.append(pred)


            #Change batch_prediction and batch into tensor data structure
            batch_prediction_tensor = torch.cat(batch_prediction, dim=0)  # Shape [batch_size, 1]
            batch_targets_tensor = torch.tensor(batch, dtype=torch.float32).reshape(-1, 1)

            #loss calculation ; l2 + base
            batch_base_loss_mse = criterion(batch_prediction_tensor, batch_targets_tensor) * alpha
            batch_base_loss_mape = mape(batch_prediction_tensor, batch_targets_tensor)* (1-alpha)
            l2_loss = sum(param.pow(2).sum() for param in model.parameters())
            loss = batch_base_loss_mse + batch_base_loss_mape  + l2_loss* l2_lambda

            #Update part
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            #Recording
            epoch_loss+=loss.item()
            all_time_loss.append(loss.item())

            batch_np = batch_prediction_tensor.detach().cpu().numpy()
            prediction = batch_np if prediction is None else np.vstack((prediction, batch_np))


        if epoch_loss< best_val_loss:
            best_val_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter+=1
        per_epoch_loss.append(epoch_loss)
        scheduler.step(epoch_loss)

    return  model, all_time_loss, per_epoch_loss, prediction

def train_loop_save(split_ratio,hidden_size, input_size,   epochs: int=5, batch_size: int = 32, lr: float = 0.0001, l2_lambda: float = 0.01,  patience: int = 5, alpha = 0.5, folder_name= None):
    model = SimpleBaseLine(hidden_size, input_size)
    data, old_mean, old_std = dataloader()
    training_data, testing_data = split_data(data= data, train_ratio = split_ratio)
    model, all_time_loss, per_epoch_loss, prediction = train_loop(model,training_data, epochs, batch_size, lr, l2_lambda, input_size, patience, alpha)
    loss, prediction  = evaluate(model, data, batch_size, input_size, l2_lambda)


    predicted_target_numpy = []
    for phase_list in prediction:
        scalar_values = [t.item() for t in phase_list]
        predicted_target_numpy.append(np.array(scalar_values))
    predicted_target_numpy = np.array([x[0] for x in predicted_target_numpy])



    if not isinstance(data, np.ndarray):
        if hasattr(data, 'values'):
            data = data.values
        else:
            data = np.array(data)

    if folder_name is None:
        folder_name = "Training_Run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(folder_name, exist_ok=True)

    predicted_target_numpy = denormalize(predicted_target_numpy, old_mean, old_std)
    data = denormalize(data, old_mean, old_std)

    # Step 8: Call your TPG plot saver function to save graphs
    run_TPG_link(
        predictions=predicted_target_numpy,
        actual=data,
        target="MPN5P",
        folder_name=folder_name
    )

    print(f"Training complete. Plots saved in folder: {folder_name}")



    pass

def move_on(batch, window, i):
    """
    In the i-th step:
    - Remove the first `min(i, len(window))` elements from the current window
    - Append `min(i, len(batch))` new elements from the batch
    Ensures the resulting window has the same fixed length.

    Handles edge cases when batch is longer than window or i > len(window).
    """
    window = torch.tensor(window, dtype=torch.float32)
    batch = torch.tensor(batch, dtype=torch.float32)

    if i == 0:
        return window

    # Determine how many to shift
    shift = min(i, len(window))
    insert = min(i, len(batch))

    # New part to append
    appended = batch[:insert]

    # Handle corner case if appended is longer than window
    if len(appended) >= len(window):
        return appended[-len(window):]  # Take last 'window size' values

    # Remove from window, append from batch
    new_window = torch.cat([window[shift:], appended])

    # If underfilled, pad front with zeros
    if len(new_window) < len(window):
        pad_len = len(window) - len(new_window)
        new_window = torch.cat([torch.zeros(pad_len), new_window])

    return new_window

def objective(trial, RFO_data):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    l2_lambda = trial.suggest_float("l2_lambda", 0.9, 0.999)
    input_size = trial.suggest_categorical("input_size", [4, 8, 16])
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256, 512])
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    alpha = trial.suggest_float("alpha", 0.0, 1.0)

    model = SimpleBaseLine(hidden_size, input_size)
    val_losses =[]

    #Entering the Data Mess
    train_val_pairs = rolling_forecast_origin_split(RFO_data)
    for train_val_pair in train_val_pairs:
        train_data, val_data = train_val_pair
        model, all_time_loss, per_epoch_loss, prediction = train_loop_save(model= model, alpha=alpha,
                                                                           epochs=10, lr= lr,
                                                                           data=train_data
                                                                           , batch_size=batch_size,
                                                                           l2_lambda=l2_lambda,
                                                                           window_size=input_size)
        val_loss , _= evaluate(model,val_data=val_data ,batch_size=batch_size , l2_lambda =l2_lambda , window_size=input_size)
        val_losses.append(val_loss)

    avg_val_loss = sum(val_losses) / len(val_losses)
    return avg_val_loss

def train_predict(split_ratio = 0.7, n_trials=25, folder_name = None):

    #Data Headache:
    data, old_mean, old_std = dataloader()
    RFO_data , test_data = split_data(data, split_ratio)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, RFO_data=RFO_data), n_trials=n_trials)
    best_params = study.best_trial.params
    print("Best Hyperparameters:", best_params)

    # Use best params to train on full RFO_data
    model = SimpleBaseLine(best_params['hidden_size'], best_params['input_size'])

    model, _, _, _ = train_loop_save(
        model=model,
        alpha=best_params['alpha'],
        epochs=10,
        lr=best_params['lr'],
        data=RFO_data,
        batch_size=best_params['batch_size'],
        l2_lambda=best_params['l2_lambda'],
        window_size=best_params['input_size']
    )

    # Final evaluation on test set
    test_loss, test_predictions = evaluate(
        model=model,
        val_data=data,
        batch_size=best_params['batch_size'],
        l2_lambda=best_params['l2_lambda'],
        window_size = best_params['input_size']
    )

    #---------------------------------------------------------------------------
    #This part of the code ; I don't know what it does ;  Package from old code
    predicted_target_numpy = []
    for phase_list in test_predictions:
        scalar_values = [t.item() for t in phase_list]
        np_array = np.array(scalar_values)
        predicted_target_numpy.append(np_array)
    predicted_target_numpy = np.array([x[0] for x in predicted_target_numpy])
    # ---------------------------------------------------------------------------

    #TPG Signature
    actual_data = denormalize(data= data,old_mean=old_mean,old_std=old_std)

    #Data Fix :
    if not isinstance(actual_data, np.ndarray):
        if hasattr(actual_data, 'values'):  # Pandas Series or DataFrame
            actual_data = actual_data.values
        else:
            actual_data = np.array(actual_data)

    predicted_target_numpy = denormalize(data= predicted_target_numpy,old_mean=old_mean,old_std=old_std)
    folder_name = folder_name if folder_name is not None else "TPG_Results"
    # Call your helper to generate and save plots:
    run_TPG_link(
        predictions=predicted_target_numpy,
        actual=actual_data,
        target="MPN5P",
        folder_name=folder_name
    )

def evaluate(model, val_data, batch_size, window_size, l2_lambda):
    print(f"Evaluating ")

    model.eval()
    with torch.no_grad():

        loss = 0
        prediction = None
        criterion = nn.MSELoss()

        batch_window_tuple_list = batch_window_generator(val_data, batch_size=batch_size, window_size=window_size)
        for batch_window_tuple in batch_window_tuple_list:
            batch, window = batch_window_tuple

            if len(batch) == 0:
                continue

            batch = list(batch)  # Ensure indexable in case it's a NumPy array
            batch_prediction = []


            for i in range(len(batch)):
                input_window = move_on(batch, window, i)
                input_window = input_window.unsqueeze(0)
                pred = model(input_window)
                batch_prediction.append(pred)

            # Change batch_prediction and batch into tensor data structure
            batch_prediction_tensor = torch.cat(batch_prediction, dim=0)  # Shape [batch_size, 1]
            batch_targets_tensor = torch.tensor(batch, dtype=torch.float32).reshape(-1, 1)

            # loss calculation ; l2 + base
            batch_base_loss = criterion(batch_prediction_tensor, batch_targets_tensor)
            l2_loss = sum(param.pow(2).sum() for param in model.parameters())
            batch_loss = batch_base_loss + l2_loss * l2_lambda

            #Recording
            loss += batch_loss.item()
            batch_np = batch_prediction_tensor.detach().cpu().numpy()
            prediction = batch_np if prediction is None else np.vstack((prediction, batch_np))

        # print(f"The target data has length of {len(val_data)}")
        # print(f"The predicted data has length od {len(prediction)}")
        # plt.plot(val_data, label ='target')
        # _ , old_mean, old_std = dataloader()
        # plt.plot(denormalize(prediction, old_mean= old_mean, old_std= old_std), label='prediction')
        # plt.show()



    return loss, prediction

def batch_window_generator(data, batch_size, window_size):

    list_batch_window_tuple = []
    total_samples = len(data)

    # Convert pandas DataFrame/Series to numpy array if needed
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.to_numpy()
    elif not isinstance(data, np.ndarray):
        raise TypeError(f"Unsupported data type: {type(data)}")

    for i in range(0, total_samples, batch_size):
        batch = data[i: i + batch_size]

        # Construct the window
        if i >= window_size:
            window = data[i - window_size: i]
        else:
            window = np.zeros(window_size)

        list_batch_window_tuple.append((batch, window))

    return list_batch_window_tuple
