
import pandas as pd
import numpy as np
import os
import math
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
# return a vector of the different data subsets as pandas DataFrames, indexed
# in numerically increasing order based on the class value
# (TODO: maybe index should == classval ?)
def split(dataset, feature, threshold):
    # decide if continuous or nominal
    data_vec = list()
    if "{n}" in feature:   # nominal
        feature_vals = dataset[feature].unique()        # obtain unique ft vals
        feature_vals.sort()                             # sort in place

        # if there's only one possible feature value, then this split won't
        # actually do any 'splitting', so return None and try different split
        if len(feature_vals) < 2:
            return None

        # otherwise, split up data based on value of given feature!
        for val in feature_vals:
            data_vec.append(dataset.loc[dataset[feature] == val])
            #TODO: make sure this guarantees going in order (i think it does)

    else:               # continuous, use threshold split
        # the following ensures no empty dataframes are returned
        less = dataset.loc[dataset[feature] < threshold]
        if len(less):
            data_vec.append(less)
        else:
            return None

        greater_eq = dataset.loc[dataset[feature] >= threshold]
        if len(greater_eq):
            data_vec.append(greater_eq)
        else:
            return None

    return data_vec


# calculate the information gain of a given split dataset
#
# data is given as a vector of pandas DataFrames so that both continuous and
# nominal data can be handled the same way
#
# also pass in the feature that was split on?  maybe not for now...
def inform_gain(data_vec):
    # conditional information gain needs the data entropy and entropy given a
    # certain feature value

    # entropy needs the probabilities of each outcome value ocurring:
    total_rows = 0.0
    for subframe in data_vec:
        total_rows += len(subframe.index)       # now total_rows = total # data

    p_target_vals = list()
    total_index = np.concatenate([(sub.index) for sub in data_vec])
    unique_target_vals, target_val_freq = np.unique(total_index,
            return_counts=True)

    # at this point, unique_target vals has all possible class vals, and
    # target_val_freq has the frequency each value occurs
    for n in range(0,len(unique_target_vals)):
        p_target_vals.append(target_val_freq[n] / sum(target_val_freq))

    if (p_target_vals[0] == 1):
        raise Exception("Invalid Split: All data resides in one subgroup")

    # cool, so we have a vector of the probabilities for which each class occurs

    # now calculate the entropy using sum(-P(y)log.2.(P(y)) for all vals
    entropy = 0
    for p in p_target_vals:
        entropy += -(p)*math.log(p,2)

    # cool, we have the entropy.  now let's find the conditional entropy and
    # subtract this from the entropy to find the conditional information gain
    # for splitting on this feature!


    # conditional entropy requires the probability that a certain feature takes
    # on each of its possible values and the conditional probability of the
    # class value given that the feature takes on each value

    conditional_entropy = 0
    for n, subframe in enumerate(data_vec):
        subframe_size = len(subframe.index)
        subframe_prob = subframe_size / total_rows

        conditional_prob = 0
        for val in [x for x in unique_target_vals if x in subframe.index]:
            val_freq = len([x for x in subframe.index == val if x == True])
            p = val_freq / subframe_size
            print("p for {} = {}".format(val, p))     # debugging
            conditional_prob += -p*math.log(p,2)

        conditional_entropy += subframe_prob*conditional_prob

        ## for debugging
        print("Conditional Prob for Feature Val {}: {} ".format(
            n, subframe_prob*conditional_prob))
        ## end debugging

    # great!  now we have both the entropy of the whole dataset given and the
    # conditional entropy of the dataset given a certain feature value.  The
    # conditional information gain is the difference of the two, so return:
    inform_gain = entropy - conditional_entropy

    ##  more debugging-ugging
    print("Unique target class values: {}".format(unique_target_vals))
    for n,p in enumerate(p_target_vals):
        print("Prob for class val {}: {}".format(unique_target_vals[n],p))
    print("Entropy in data: {}".format(entropy))

    print("Conditional Entropy: {}".format(conditional_entropy))
    print("Information Gain given this feature and split: {}".format(
                inform_gain))
    ## end debugging

    if (inform_gain >= 0):
        return inform_gain
    else:
        raise Exception("Something went wrong... Information Gain is negative")


# iterate through all possible splits on the dataset provided and return the
# split with lowest information gain
#
# only input needed is the dataset to split
#
# output is the feature to split on, best threshold for cont. features, and a
# data vector containing the split data
def calc_best_split(dataset):
    feature, threshold, inf_gain, data_vec = None, None, 100, list()
    for col in dataset.columns:
        if "{n}" in col:    # nominal feature
            data_vec_temp = split(dataset, col, t)
            if data_vec_temp == None:
                continue

            inf_gain_temp = inform_gain(data_vec_temp)
            print("Feature: {}\nInf_Gain: {}\n\n".format(
                feature, inf_gain_temp))

            if inf_gain_temp < inf_gain:
                inf_gain = inf_gain_temp
                data_vec = data_vec_temp
                feature = col
                threshold = t
            else:
                continue

        else:               # continuous feature
            for t in dataset[col].unique():
                data_vec_temp = split(dataset, col, t)
                if data_vec_temp == None:
                    continue

                inf_gain_temp = inform_gain(data_vec_temp)
                print("Feature: {}\nInf_Gain: {}\nSplit: {}\n\n".format(
                    feature, inf_gain_temp, t))

                if inf_gain_temp < inf_gain:
                    inf_gain = inf_gain_temp
                    data_vec = data_vec_temp
                    feature = col
                    threshold = t
                else:
                    continue

    return feature, threshold, inf_gain, data_vec











