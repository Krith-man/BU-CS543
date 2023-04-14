import numpy as np
import pandas as pd
from collections import Counter
import utils


def artificial_dataset():
    ''' 
    Returns the artificial data from the csv file 
    '''
    data = np.loadtxt('data/artificial_data.csv', delimiter=',') 
    return data.astype(int)

if __name__ == '__main__':
    # Retrieve artificial data
    buckets = 1000
    num_hash_func = 2
    data = artificial_dataset()
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

    # Calculate the probability of overestimate items and
    # compare it with the theoretical bound
    overestimate_counter = 0
    for investigate_item in items:
        result = instance.calculate(item=investigate_item)
        deviation = result - actual_freq_dict[investigate_item]
        if deviation >= (2 * total_elements) / buckets:
            overestimate_counter += 1
    
    print("Total number of elements in the stream: {}".format(total_elements))
    print("Theoretical bound of CountMinSketch with {} buckets and {} hash functions: {}".format(
        buckets, num_hash_func, 1/(pow(2, num_hash_func))))
    print("Probability of overestimating elements: {}".format(
        overestimate_counter/len(items)))
