import pandas as pd
import os

dataset_path = r"Datasets"  
output_file = r"Datasets/nyc_taxi_sample.csv"  

df = pd.concat(
    [pd.read_csv(os.path.join(dataset_path, file), nrows=500) for file in os.listdir(dataset_path) if file.endswith(".csv")], 
    ignore_index=True
)

df.to_csv(output_file, index=False)

print("CSV files merged successfully!")


