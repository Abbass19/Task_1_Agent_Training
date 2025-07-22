import os
from Code.Version_3.defined_modules import SimpleBaseLine
from Code.Version_3.functions import split_data, train_loop_save, evaluate
from Code.Version_3.preprocessing_functions import dataloader, smooth_data, denormalize
import matplotlib.pyplot as plt


csv_path = os.path.join(os.path.dirname(__file__), "my_data.csv")
spike_frequency = 8


Non_Smoothed_Data, old_mean, old_std = dataloader(csv_path)
smoothed_data = smooth_data(Non_Smoothed_Data, window_size=5)

plt.plot(Non_Smoothed_Data, label ='Non Smoothed Data')
plt.plot(smoothed_data, label = 'Smoothed Data')
plt.show()


training_non_smooth_data, testing_non_smooth_data  = split_data(train_ratio=0.7, data=Non_Smoothed_Data)
training_smooth_data, testing_smooth_data  = split_data(train_ratio=0.7, data=smoothed_data)


model_Non_Smoothed_Data = SimpleBaseLine(64, spike_frequency)
model_Non_Smoothed_Data ,all_time_loss, per_epoch_loss , prediction = train_loop_save(model= model_Non_Smoothed_Data, data=training_non_smooth_data, epochs=20, batch_size=spike_frequency)
_ , prediction_non_smooth = evaluate(model_Non_Smoothed_Data, val_data=Non_Smoothed_Data, batch_size=spike_frequency, window_size=spike_frequency)


plt.plot(denormalize(Non_Smoothed_Data,  old_mean = old_mean, old_std= old_std), label='Non Smoothed Data')
plt.plot(denormalize(prediction_non_smooth,old_mean = old_mean, old_std= old_std), label='Non Smoothed Prediction')
plt.title("Here we should have spikes ; This is accepted")
plt.show()

model_smoothed_data = SimpleBaseLine(64, spike_frequency)
model_smoothed_data, _ , _ , _ = train_loop_save(model= model_smoothed_data, data=training_non_smooth_data, epochs=20, batch_size=spike_frequency, window_size=spike_frequency)
_ , prediction_smooth = evaluate(model_smoothed_data, val_data=smoothed_data, batch_size=spike_frequency, window_size=spike_frequency)

plt.plot(denormalize(smoothed_data, old_mean = old_mean,old_std= old_std), label=' Smoothed Data')
plt.plot(denormalize(prediction_smooth,old_mean = old_mean, old_std= old_std), label=' Prediction for smoothed data')
plt.title("Have we lost the spikes ; This is accepted")
plt.show()