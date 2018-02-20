
import pandas as pd
import os
import subprocess

# import the dataset
#
# give path to data (local or url) and column to classify on)
#
# by default, look at "heart.data" locally then remotely
def import_dataset(path = "heart.data", index_col = 13):
    if os.path.exists("heart.data"):
        print("Dataset: {}\nFound: Locally".format(path))
        try:
            df = pd.read_csv(path, header=0, engine='python', index_col=index_col)
            # note: engine selection as python is more stable on my machine
            # using c gives random segfaults
        except IndexError as ind:
            print(ind)
            print("Index provided: {}".format(index_col))
            exit()
        except:
            print("Failed to parse data set...")
            raise
    else:
        print("Dataset not found locally.\nDownloading dataset...")
        try:
            df = pd.read_csv(path)
        except:
            exit("Not a valid dataset URL")

        with open(path, 'w') as local_file:
            print("Saving dataset locally")
            df.to_csv(local_file)
    return df


# split the given dataset on the feature given
#
# specify a threshold for continuous splits, recognized by {n} or {c} in the
# feature name
#
# return a vector of the different data subsets, indexed in numerically
# increasing order based on the class value
# (TODO: maybe index should == classval ?)
def split(dataset, feature, threshold):
    # decide if continuous or nominal
    data_vec = {}
    if "{n}" in feature:   # nominal
        for val in dataset[feature].unique():
            data_vec[val] = dataset.loc[dataset[feature] == val]

    else:               # continuous, use threshold split
        data_vec["<" + str(threshold)] = dataset.loc[dataset[feature] < threshold]
        data_vec[">=" + str(threshold)] = dataset.loc[dataset[feature] >= threshold]

    return data_vec


# calculate the information gain of a given split dataset
#
# data is given as a vector of pandas Series so that both continuous and
# nominal data can be handled the same way
#
# also pass in the feature that was split on plz
def inform_gain(data_vec, feature):


    return ig_val
