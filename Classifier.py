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

    return new_df.apply(pd.to_numeric)  # convert strings to numbers

# import the dataset as a pandas dataframe
df = DecisionTreeTools.import_dataset()

# preprocess to add tag for continuous vs nominal
parsed_df = preprocess_heart_data(df)

# build decision tree
tree_root = DecisionTreeTools.build_decision_tree(parsed_df, 25, 25)

# print out the tree
DecisionTreeTools.print_tree(tree_root, 0)

## TODO: separate data into 80% training and 20% testing, then evaluate
# performance of this algorithm

