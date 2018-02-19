#/usr/bin/python

import numpy as np
import pandas as pd

import DecisionTreeTools

# preprocess the dataset by storing information regarding continuous vs nominal
# features
def preprocess_heart_data(df):
    new_columns = df.columns.values

    for n, column in enumerate(new_columns):
        new_columns[n] = df.columns.values[n] + " {" + df.iloc[0][column] + "}"

    new_df = df.drop(df.index[0])
    new_df.columns = new_columns

    return new_df


# import the dataset as a pandas dataframe
df = DecisionTreeTools.import_dataset()

# preprocess to add tag for continuous vs nominal
parsed_df = preprocess_heart_data(df)

print(parsed_df)
