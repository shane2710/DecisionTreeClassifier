
import pandas as pd
import os
import subprocess

# function for importing the dataset
#
# give path to data (local or url) and column to classify on)
#
# by default, look at "heart.data" locally then remotely
def import_dataset(path = "heart.data", index_col = 13):
    if os.path.exists("heart.data"):
        print("Dataset: {}\nFound: Locally".format(path))
        try:
            df = pd.read_csv(path, header=0, index_col=index_col)
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
