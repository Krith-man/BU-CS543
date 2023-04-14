import numpy as np
import pandas as pd
from collections import Counter
import utils
import argparse


def lastFM_asia_dataset():
    '''
    Returns the data from the csv file of dataset "LastFM Asia Social Network"
    The data includes the destination nodes of the provided graph.
    The total number of elements in our stream is: 27,806
    '''
    df = pd.read_csv("data/lastfm_asia_edges.csv")
    return df['node_2'].to_numpy()


def facebook_dataset():
    '''
    Returns the data from the csv file of dataset "Social circles: Facebook"
    The data includes the destination nodes of the provided graph.
    The total number of elements in our stream is: 88,234
    '''
    data = []
    with open('data/facebook_combined.txt') as file:
        for line in file:
            edges = line.rstrip().split(" ")
            data.append(int(edges[1]))
    return np.array(data)


def wiki_dataset():
    '''
    Returns the data from the csv file of dataset "Wikipedia vote network"
    The data includes the destination nodes of the provided graph.
    The total number of elements in our stream is: 103,689
    '''
    data = []
    with open('data/wiki-Vote.txt') as file:
        for line in file:
            edges = line.rstrip().split("\t")
            data.append(int(edges[1]))
    return np.array(data)


if __name__ == '__main__':
    # Parse the arguments provided
    parser = argparse.ArgumentParser(description='Question 5')
    parser.add_argument(
        '-d', '--dataset', help='Choose "f" for Facebook ,or "l" for LastFM Asia dataset or "w" for Wiki dataset', required=True)
    args = parser.parse_args()

    buckets = 1000
    num_hash_func = 2
    # Retrieve data from selected dataset
    if args.dataset == "f":
        print("Facebook dataset was chosen!")
        data = facebook_dataset()
    elif args.dataset == "l":
        print("LastFM Asia dataset was chosen!")
        data = lastFM_asia_dataset()
    else:
        print("Wiki dataset was chosen!")
        data = wiki_dataset()

    max_value = np.max(data)
    total_elements = len(data)

    # Add items to CountMin sketch table
    instance = utils.CountMinSketch(
        buckets=buckets, num_hash_func=num_hash_func, max_value=max_value)
    for item in data:
        instance.addItem(item=item)

    # Find actual frequency values of data items
    actual_freq_dict = dict(Counter(data))
    items = list(actual_freq_dict.keys())

    # Calculate the probability of overestimate items multiple times
    num_runs = 5
    prob = []
    for i in range(num_runs):
        overestimate_counter = 0
        for investigate_item in items:
            deviation = instance.calculate(item=investigate_item) - actual_freq_dict[investigate_item]
            if deviation >= (2 * total_elements) / buckets:
                overestimate_counter += 1
        prob.append(overestimate_counter/len(items))

    print("Total number of elements in the stream: {}".format(len(data)))
    print("Theoretical bound of CountMinSketch with {} buckets and {} hash functions: {}".format(
        buckets, num_hash_func, 1/(pow(2, num_hash_func))))
    print("Mean Probability of overestimating elements after {} runs of countMin sketch algorithm: {}".format(num_runs,
        np.mean(prob)))
