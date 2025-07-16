from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt 
import seaborn as sns
import datetime
import joblib
import sys
# from tests_folder.tests import resolve_TargetPlotsUploader
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_percentage_error
# from tests_folder import settings
plt.style.use('ggplot')

def sir_parameters(x,y):
  x=np.array(x)
  y=np.array(y)
  analytical_params = linregress(x, y)

  slope = analytical_params.slope
  intercept = analytical_params.intercept
  rvalue = analytical_params.rvalue
  y_trend_line = slope*x + intercept
  x_trend_line = x 
  avg_trend_line_distance = np.mean(np.abs(y_trend_line - y)/y_trend_line)
  accuracy = np.mean(np.abs(x_trend_line - y)/x_trend_line)

  return slope, intercept, rvalue**2, avg_trend_line_distance, accuracy,x_trend_line

def comparison_plot(predicted_targets,actual_targets,dates,target, calculated_yield):

  #1 ; predicted_target : I think this is the model's estimated stock price
  #2 ; actual_target : This should be the real price changes in the csv file
  #3 ; dates : The date for each stock price and prediction
  #4 ; target : No idea what is this
  #5 ; calculated_yield : no Idea what is this


  trend_slope,trend_intercept,trend_r2,dispersion,accuracy,_ =sir_parameters(actual_targets,predicted_targets)
  fig,ax = plt.subplots(1,figsize=(20,16))

  date_format = "%Y-%m-%d"
  dates = [datetime.datetime.strptime(date, date_format) for date in dates]
  dates = (np.array(dates)).squeeze()
  actual_targets = np.array(actual_targets)

  sns.lineplot(x=dates,y=actual_targets,ax=ax, label='Actual')
  sns.lineplot(x=dates,y=predicted_targets,ax=ax, label='Predicted')

  ax.set_xlabel('DATE', fontsize=20)
  ax.set_ylabel(f'{target}', fontsize=20)

  dates = pd.Series(pd.to_datetime(dates))
  start_date = str(dates.min())
  end_date = str(dates.max())

  plt.title(f'Comparison graph - Predicted and Actual Targets versus Dates for {target} from {start_date} to {end_date}', fontweight='bold', fontsize=20)
  plt.legend(fontsize=20)
  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)

  text_size = 'x-large'

  plt.figtext(.92,.85,['trend_slope:',round(trend_slope,4)], fontsize = text_size)
  plt.figtext(.92,.80,['trend_intercept:',round(trend_intercept,4)], fontsize=text_size)
  plt.figtext(.92,.75,['trend_r2:',round(trend_r2,4)], fontsize=text_size)
  plt.figtext(.92,.70,['trend_standard_deviation:',round(np.std(actual_targets),4)], fontsize=text_size)
  plt.figtext(.92,.65,['trend_dispersion:',round(dispersion,4)], fontsize = text_size)
  plt.figtext(.92,.60,['accuracy:',round(accuracy,4)], fontsize = text_size)
  plt.figtext(.92,.55,['yield:',round(calculated_yield,4)], fontsize = text_size)


  temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
  plt.savefig(temp_file.name,  bbox_inches='tight')
  plt.close()
  result = {
    'trend_slope' : trend_slope,
    'trend_intercept': trend_intercept,
    'trend_r2': trend_r2,
    'dispersion': dispersion
  }

  return result,temp_file.name

def ratio_plot(predicted_targets,actual_targets,dates,target, calculated_yield):

  #predicted_target
  #actual_targer
  # dates
  #target
  #


  predicted_targets = np.array(predicted_targets)
  actual_targets = np.array(actual_targets)
  fig,ax = plt.subplots(1,figsize=(20,16))
  ratio = predicted_targets/actual_targets
  a=np.linspace(0,len(actual_targets),len(actual_targets),dtype=np.int32)
  z=np.polyfit(a,ratio,1)
  p=np.poly1d(z)

  plt.plot(ratio,label='Ratio') 
  plt.plot(a,p(a), alpha=0.75, label='Fitted Line')

  plt.xlabel('DATE',fontsize=20)
  plt.ylabel(f'{target}', fontsize=20)
  dates = pd.Series(pd.to_datetime(dates))
  start_date = str(dates.min())
  end_date = str(dates.max())
  plt.title(f' Ratio plot - Predicted / Actual Targets Ratios versus Dates for {target} from {start_date} to {end_date}', fontweight='bold', fontsize=20)
  plt.legend(fontsize=20)
  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)

  trend_slope,trend_intercept,trend_r2,dispersion,accuracy,_ =sir_parameters(actual_targets,ratio)
  text_size = 'x-large'
  plt.figtext(.92,.85,['trend_slope:',round(trend_slope,4)], fontsize = text_size)
  plt.figtext(.92,.80,['trend_intercept:',round(trend_intercept,4)], fontsize=text_size)
  plt.figtext(.92,.75,['trend_r2:',round(trend_r2,4)], fontsize=text_size)
  plt.figtext(.92,.70,['trend_standard_deviation:',round(np.std(actual_targets),4)], fontsize=text_size)
  plt.figtext(.92,.65,['trend_dispersion:',round(dispersion,4)], fontsize = text_size)
  plt.figtext(.92,.60,['yield:',round(calculated_yield,4)], fontsize = text_size)
  # plt.tight_layout()  # Adjust subplots parameters
  temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
  plt.savefig(temp_file.name,  bbox_inches='tight')
  plt.close()
  # plt.savefig(plot_path + '_ratio',  bbox_inches='tight')

  result = {
    'trend_slope' : trend_slope,
    'trend_intercept': trend_intercept,
    'trend_r2': trend_r2,
    'dispersion': dispersion
  }

  return result,temp_file.name

def scatter_plot(predicted_targets,actual_targets,dates,target, calculated_yield):
  trend_slope,trend_intercept,trend_r2,dispersion,accuracy, _ =sir_parameters(actual_targets,predicted_targets)

  fig,ax = plt.subplots(1,figsize=(20,16))

  sns.regplot(x=actual_targets,y=predicted_targets,ax=ax)

  ax.set_xlabel(f'Actual Target {target}', fontsize=20)
  ax.set_ylabel(f'Predicted Target {target}', fontsize=20)
  dates = pd.Series(pd.to_datetime(dates))

  start_date = str(dates.min())
  end_date = str(dates.max())
  plt.title(f'Scatter graph - Predicted and Actual Targets versus Dates for {target} from {start_date} to {end_date}', fontweight='bold', fontsize=15)
  text_size = 'x-large'

  plt.figtext(.92,.85,['trend_slope:',round(trend_slope,4)], fontsize = text_size)
  plt.figtext(.92,.80,['trend_intercept:',round(trend_intercept,4)], fontsize=text_size)
  plt.figtext(.92,.75,['trend_r2:',round(trend_r2,4)], fontsize=text_size)
  plt.figtext(.92,.70,['trend_standard_deviation:',round(np.std(actual_targets),4)], fontsize=text_size)
  plt.figtext(.92,.65,['trend_dispersion:',round(dispersion,4)], fontsize = text_size)
  plt.figtext(.92,.60,['yield:',round(calculated_yield,4)], fontsize = text_size)
  # plt.tight_layout()  # Adjust subplots parameters
  temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
  plt.savefig(temp_file.name,  bbox_inches='tight')
  plt.close()
  # plt.savefig(plot_path + '_scatter',  bbox_inches='tight')

  result = {
      'trend_slope' : trend_slope,
      'trend_intercept': trend_intercept,
      'trend_r2': trend_r2,
      'dispersion': dispersion
  }

  return result,temp_file.name

def accuracy_plot(predicted_targets,actual_targets,dates,target, calculated_yield):
    trend_slope,trend_intercept,trend_r2,dispersion,accuracy,x_trend_line=sir_parameters(actual_targets,predicted_targets)
    fig,ax = plt.subplots(1,figsize=(20,16))
    plt.scatter(actual_targets, predicted_targets)
    plt.plot(actual_targets, x_trend_line, color = 'green')
    ax.set_xlabel(f'Actual Target {target}', fontsize=20)
    ax.set_ylabel(f'Predicted Target {target}', fontsize=20)
    dates = pd.Series(pd.to_datetime(dates))

    start_date = str(dates.min())
    end_date = str(dates.max())
    plt.title(f'Scatter graph - Predicted and Actual Targets versus Dates for {target} from {start_date} to {end_date}', fontweight='bold', fontsize=15)
    text_size = 'x-large'
    plt.figtext(.92,.85,['trend_slope:',1], fontsize = text_size)
    plt.figtext(.92,.80,['trend_intercept:',0], fontsize=text_size)
    plt.figtext(.92,.75,['trend_r2:',1], fontsize=text_size)
    # plt.figtext(.92,.70,['trend_standard_deviation:',round(np.std(actual_targets),4)], fontsize=text_size)
    plt.figtext(.92,.70,['accuracy:',round(accuracy,4)], fontsize = text_size)
    plt.figtext(.92,.65,['yield:',round(calculated_yield,4)], fontsize = text_size)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name,  bbox_inches='tight')
    # plt.savefig('/learn/comparision.png',  bbox_inches='tight')
    plt.close()
    result = {
      'trend_slope' : trend_slope,
      'trend_intercept': trend_intercept,
      'trend_r2': trend_r2,
      'dispersion': dispersion
    }
    
    return result,temp_file.name

def generate_plot(predicted_targets,actual_targets,dates,target, calculated_yield):
  print("Started")
  predicted_targets = np.array(predicted_targets)
  actual_targets = np.array(actual_targets)
  # dates = np.array(dates,dtype='datetime64')

  plot_funcs = {
    'Comparision': comparison_plot(predicted_targets,actual_targets,dates,target, calculated_yield)[0],
    'Comparisio_plot_address':comparison_plot(predicted_targets,actual_targets,dates,target, calculated_yield)[1],
    'Ratio': ratio_plot(predicted_targets,actual_targets,dates,target, calculated_yield)[0],
    'Ratio_Plot_address': ratio_plot(predicted_targets,actual_targets,dates,target, calculated_yield)[1],
    'Scatter': scatter_plot(predicted_targets,actual_targets,dates,target, calculated_yield)[0],
    'Scatter_plot_address':scatter_plot(predicted_targets,actual_targets,dates,target, calculated_yield)[1],
    'Accuracy': accuracy_plot(predicted_targets,actual_targets,dates,target, calculated_yield)[0],
    'Scatter_plot_address':  accuracy_plot(predicted_targets,actual_targets,dates,target, calculated_yield)[1]
  }
  # result = {}
  # plot_type ={}
  # for value in plot_funcs.values():
  #  result[value] = plot_funcs[value](predicted_targets,actual_targets,dates,target)[1]
  # # results_list = [value for value in plot_funcs.values()]
  # # print(results_list)
  # for value in plot_funcs.keys():
  #  plot_type[value] = plot_funcs[value](predicted_targets,actual_targets,dates,target)
  # # plot_type = [value for value in plot_funcs.keys()]

  return plot_funcs

def image_addresses(predicted_targets,actual_targets,dates,target, calculated_yield):
  # print("started")
  result,comparison_plot_address = comparison_plot(predicted_targets,actual_targets,dates,target, calculated_yield)
  result,ratio_plot_address = ratio_plot(predicted_targets,actual_targets,dates,target, calculated_yield)
  result,scatter_plot_address = scatter_plot(predicted_targets,actual_targets,dates,target, calculated_yield)
  result,accuracy_plot_address = accuracy_plot(predicted_targets,actual_targets,dates,target, calculated_yield)
  # print("ok")
  plot_addresses = {
    'comparison_plot': comparison_plot_address,
    'ratio_plot': ratio_plot_address,
    'scatter_plot': scatter_plot_address,
    'accuracy_plot_address': accuracy_plot_address
  }
  return plot_addresses

