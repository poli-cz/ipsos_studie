# open file batch_500.csv and load it into pandas dataframe

import pandas as pd

df = pd.read_csv("batch_500.csv")


# compute statistics for each atribut for residence column

residence = df["political_preferences"]

# dict for storing residence and count
residence_dict = {}

for i in residence:
    if i in residence_dict:
        residence_dict[i] += 1
    else:
        residence_dict[i] = 1


# recalculate to percentage
total_residence = len(residence)
for key in residence_dict:
    residence_dict[key] = residence_dict[key] / total_residence * 100

print(residence_dict)
