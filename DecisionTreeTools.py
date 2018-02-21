
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
# TODO: should have a more portable way of deciding nominal vs cont, maybe
# threshold = None for nominal?
def train_split(dataset, feature, threshold):
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
        #raise Exception("Invalid Split: All data resides in one subgroup")
        return 0        # this split was no good, doesn't actually split any
                        # data, so return 0 - we gain nothing!

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
            #print("p for {} = {}".format(val, p))     # debugging
            conditional_prob += -p*math.log(p,2)

        conditional_entropy += subframe_prob*conditional_prob

        ## for debugging
        #print("Conditional Prob for Feature Val {}: {} ".format(
        #    n, subframe_prob*conditional_prob))
        ## end debugging

    # great!  now we have both the entropy of the whole dataset given and the
    # conditional entropy of the dataset given a certain feature value.  The
    # conditional information gain is the difference of the two, so return:
    inform_gain = entropy - conditional_entropy

    ###  more debugging-ugging
    #print("Unique target class values: {}".format(unique_target_vals))
    #for n,p in enumerate(p_target_vals):
    #    print("Prob for class val {}: {}".format(unique_target_vals[n],p))
    #print("Entropy in data: {}".format(entropy))

    #print("Conditional Entropy: {}".format(conditional_entropy))
    #print("Information Gain given this feature and split: {}".format(
    #            inform_gain))
    ### end debugging

    if (inform_gain >= 0):
        return inform_gain
    else:
        raise Exception("Something went wrong... Information Gain is negative")


# process a terminal node once one of the terminal conditions is hit
#
# input is the dataset that exists at the current node
#
# output is the most likely class value
def terminate_node(dataset):
    likely_class = None
    freq = 0
    for class_val in dataset.index.unique():
        if len([x for x in dataset.index == class_val if x == True]) > freq:
            likely_class = class_val

    # note: ran into a small issue: when multiple class vals are the same
    # frequency and are tied for first, always returns same val w/o warning
    return likely_class


# iterate through all possible splits on the dataset provided and return the
# split with lowest information gain
#
# only input needed is the dataset to split
#
# output is dict: feature to split on, best threshold for cont. features, and a
# data vector containing the split data
def calc_best_split(dataset):
    feature, threshold, inf_gain, data_vec = None, np.array([0]), 0, None
    for col in dataset.columns:
        #print(col)
        if "{n}" in col:    # nominal feature
            data_vec_temp = train_split(dataset, col, t)
            if data_vec_temp == None:
                continue

            inf_gain_temp = inform_gain(data_vec_temp)
            #print("Feature: {}\nInf_Gain: {}\n\n".format(
                #feature, inf_gain_temp))

            if inf_gain_temp > inf_gain:
                inf_gain = inf_gain_temp
                data_vec = data_vec_temp
                feature = col
                feature_vals = dataset[col].unique()
                feature_vals.sort()
                threshold = feature_vals
            else:
                continue

        else:               # continuous feature
            for t in dataset[col].unique():
                data_vec_temp = train_split(dataset, col, t)
                if data_vec_temp == None:
                    continue

                inf_gain_temp = inform_gain(data_vec_temp)
                #print("Feature: {}\nInf_Gain: {}\nSplit: {}\n\n".format(
                    #feature, inf_gain_temp, t))

                if inf_gain_temp > inf_gain:
                    inf_gain = inf_gain_temp
                    data_vec = data_vec_temp
                    feature = col
                    threshold[0] = t
                else:
                    continue

    if feature == None:
        # we didn't find a feature to split on, so this must be a terminal node
        return terminate_node(dataset)
    else:
        # otherwise this is the best split
        return {'feature':feature, 'value':threshold, 'data':data_vec}



# recursive splitting function used to split on them splits
#
# inputs are current tree depth, minimum data size, max depth, and current node
#
# outputs are nothing, just add child_num:child_node key:val pairs to current
# node and move the data along to those children
def recursive_split(node, max_depth, min_size, current_depth):

    print("Feature: {}\tValues: {}\tData Len: {}".format(node['feature'],
            node['value'], len(node['data'])))

    # move data from current node into temp list
    children_data = list()
    for data in node['data']:
        children_data.append(data)

    del(node['data'])

    # generate placeholder for vector of children nodes
    num_children = len(children_data)
    node['children'] = {}

    print("Depth: {}\tChildren: {}".format(current_depth, num_children))
    # check for max depth of the tree
    if current_depth >= max_depth:
        for n in range(num_children):
            node['children'][node['value'][n]] = terminate_node(children_data[n])
        return

    # otherwise process each child
    if len(node['value']) > 1:          # nominal feature
        for child in range(num_children):
            if len(children_data[child]) > min_size:
                node['children'][node['value'][child]] = calc_best_split(
                        children_data[child])
                if type(node['children'][node['value'][child]]) == dict:
                    recursive_split(node['children'][node['value'][child]],
                            max_depth, min_size, current_depth+1)
            else:
                node['children'][node['value'][child]] = terminate_node(
                        children_data[child])
    else:
        # continuous feature, process 'less' then 'greater'
        if len(children_data[0]) > min_size:
            node['children']['less'] = calc_best_split(children_data[0])
            if type(node['children']['less']) == dict:  # got a valid node back
                recursive_split(node['children']['less'],
                        max_depth, min_size, current_depth+1)
            else:           # must've been a terminal node
                node['children']['less'] = terminate_node(children_data[0])
        else:
            node['children']['less'] = terminate_node(children_data[0])

        if len(children_data[1]) > min_size:
            node['children']['greater'] = calc_best_split(children_data[1])
            if type(node['children']['greater']) == dict:
                recursive_split(node['children']['greater'],
                        max_depth, min_size, current_depth+1)
            else:           # must've been a terminal node
                node['children']['greater'] = terminate_node(children_data[1])
        else:
            node['children']['greater'] = terminate_node(children_data[1])

    return

    ## note: setup termination correctly, then name children nodes by their
    # values if discrete.  probably need a switch statement for handling
    # cont vs nom values differently through the above!
    ### then, i think it's just processing each node in a loop, and good to go?




# it's time to build the tree!!!
#
# inputs include base dataset, max depth of tree, and minimum data size @ node
#
# output is the root node of the tree!
def build_decision_tree(dataset, max_depth, min_size):

    # first get root of tree
    root = calc_best_split(dataset)

    # now call the recursive splitting function to construct the rest!
    recursive_split(root, max_depth, min_size, 1)

    return root



# print out the tree for visualizationz!
#
# inputs are starting node for visualizing and the depth
#
# returns nothing but outputs a cool indented hierarchy of nodes on terminal!
def print_tree(node, depth):
    if isinstance(node, dict):
        if "{n}" in node['feature']:        # nominal feature
            for n,child in enumerate(node['children']):
                print('%s[%s = %.3f]' % ((depth*' ',
                    (node['feature']),node['value'][n])))
                print_tree(node['children'][child], depth+1)
        else:                               # continuous feature
            print('%s[%s < %.3f]' % ((depth*' ',
                (node['feature']),node['value'][0])))
            for child in node['children']:
                print_tree(node['children'][child], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))






