import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import utils
import argparse

NUM_HASH_FUNCTIONS = 4


def artificial_dataset():
    ''' 
    Returns the data from the csv file of randomly selected data points.
    '''
    data = np.loadtxt('data/artificial_data.csv', delimiter=',')
    return data.astype(int)


def real_dataset():
    ''' 
    Returns the data from the csv file of dataset "LastFM Asia Social Network"
    The data includes the destination nodes of the provided graph.
    The total number of elements in our stream is: 27,806
    '''
    df = pd.read_csv("data/lastfm_asia_edges.csv")
    return df['node_2'].to_numpy()


def mean_deviation(instance, actual_freq_dict, data):
    deviation = []
    # Add items to countMin sketch
    for item in data:
        instance.addItem(item)
    # Calculate the mean deviation per item
    for key, value in actual_freq_dict.items():
        deviation.append(instance.calculate(key) - float(value))
    return np.mean(deviation)


if __name__ == '__main__':
    # Parse the arguments provided
    parser = argparse.ArgumentParser(description='Question 4')
    parser.add_argument(
        '-d', '--dataset', help='Choose "a" for artificial or "r" for real dataset', required=True)
    args = parser.parse_args()

    # Retrieve data from selected dataset
    if args.dataset == "a":
        print("Artifical dataset was chosen!")
        data = artificial_dataset()
    else:
        print("Real dataset was chosen!")
        data = real_dataset()
    max_value = np.max(data)
    total_elements = len(data)
    # Find actual frequency values of data items
    actual_freq_dict = dict(Counter(data))
    items = list(actual_freq_dict.keys())

    # Plot performance(mean error) of countMin sketch algorithm
    # with various K(number of buckets) and 5 hash functions.
    NUM_BUCKETS = 200
    error = []
    for k in range(1, NUM_BUCKETS):
        instance = utils.CountMinSketch(
            buckets=k, num_hash_func=NUM_HASH_FUNCTIONS, max_value=max_value)
        error.append(mean_deviation(instance, actual_freq_dict, data))
    plt.plot(range(1, NUM_BUCKETS), error)
    plt.xlabel('# Buckets')
    plt.ylabel('Mean Deviation')
    plt.title('Mean Deviation of CountMin Sketch over Different Size of Buckets \n # Buckets = {}'.format(
        NUM_BUCKETS))
    if args.dataset == "a":
        plt.savefig('plots/q4/' + 'Artifical Dataset: ' +
                    'K= ' + str(NUM_BUCKETS) + '.png')
    else:
        plt.savefig('plots/q4/' + 'Real Dataset: ' +
                    'K= ' + str(NUM_BUCKETS) + '.png')
    plt.clf()

    # Plot probability error of countMin sketch algorithm
    # with various N(number of hash functions) and 1000 buckets.
    NUM_BUCKETS = 1000
    prob = []
    for n in range(1, NUM_HASH_FUNCTIONS+1):
        instance = utils.CountMinSketch(
            buckets=NUM_BUCKETS, num_hash_func=n, max_value=max_value)
        overestimate_counter = 0
        for item in data:
            instance.addItem(item=item)
        for investigate_item in items:
            deviation = instance.calculate(
                item=investigate_item) - actual_freq_dict[investigate_item]
            if deviation >= (2 * total_elements) / NUM_BUCKETS:
                overestimate_counter += 1
        prob.append(overestimate_counter/len(items))
    plt.plot(range(1, NUM_HASH_FUNCTIONS+1), prob)
    plt.xlabel('# Hash Functions')
    plt.ylabel('Error Probability')
    plt.xticks(range(1, NUM_HASH_FUNCTIONS+1))
    plt.title('Error Probability of CountMin Sketch over Different Number of Hash Functions \n # Hash Functions = {}'.format(
        NUM_HASH_FUNCTIONS))
    if args.dataset == "a":
        plt.savefig('plots/q4/' + 'Artifical Dataset: ' +
                    'N= ' + str(NUM_HASH_FUNCTIONS) + '.png')
    else:
        plt.savefig('plots/q4/' + 'Real Dataset: ' +
                    'N= ' + str(NUM_HASH_FUNCTIONS) + '.png')
