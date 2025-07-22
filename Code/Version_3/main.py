from Code.Version_3.preprocessing_functions import analyze_distribution
from Code.Version_3.targets_plot_generator import save_plots_to_default_folder
from matplotlib import pyplot as plt
from functions import *
import os


# train_ratio = 0.7
# #Some Storage Data
# csv_path = os.path.join(os.path.dirname(__file__), "my_data.csv")
# data, old_mean, old_std = dataloader(csv_path)
#
#
#
# # fig , axis = plt.subplots(1,2)
# # axis[0].plot(denormalize(data, old_mean=old_mean, old_std= old_std))
# # axis[0].set_title('This is data Before Preprocessing')
# #
# # axis[1].plot(data)
# # axis[1].set_title('This is data after Preprocessing')
# # plt.tight_layout()
# # plt.show()
# #
# #
#
#
#
#
#
# training_data, testing_data  = split_data(train_ratio=train_ratio, data=data)
#
#
# model = SimpleBaseLine(64,8)
# model ,all_time_loss, per_epoch_loss , prediction = train_loop_save(model= model, training_data=training_data, epochs=40, batch_size=16, window_size=8, alpha=0.274)
#
#
# plt.plot(all_time_loss)
# plt.title("Loss during Training for All data")
# plt.show()
#
# plt.plot(denormalize(data,old_mean, old_std), label= ' Real Data')
# plt.plot(denormalize(prediction,old_mean, old_std), label = ' Predicted Data')
# plt.title('Comparison of Two Plots of Training Only')
# plt.xlabel("Time")
# plt.ylabel("Stock - Price")
# plt.show()
#
#
# loss , prediction_2 = evaluate(model, val_data=data, l2_lambda=0.988, window_size=8, batch_size=16)
#
#
# #Conctruct the tpg signature:
#
#
#
# plt.plot(denormalize(data,old_mean, old_std), label= ' Real Data')
# plt.plot(denormalize(prediction_2,old_mean, old_std), label = ' Predicted Data')
# plt.title('Comparison of Two Plots of Evaluate Function')
# plt.xlabel("Time")
# plt.ylabel("Stock - Price")
# plt.show()


# train_predict(0.7, 1)

train_loop_save(split_ratio=0.7, hidden_size=64, input_size=8, epochs=40, batch_size=16,
                lr=0.000993, l2_lambda=0.007205, patience=5, alpha=0.274,folder_name="64_")

