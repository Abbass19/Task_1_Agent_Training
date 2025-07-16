#Some Storage Data
import os
import pandas as pd

csv_path = os.path.join(os.path.dirname(__file__), "models", "my_data.csv")
data = pd.read_csv(csv_path)

actual_target = data.loc[:, 'MPN5P']
actual_target = actual_target.to_numpy()

print(actual_target)
