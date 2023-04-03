import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import utils


def real_dataset():
    ''' 
    Returns the real data from the csv file of dataset "LastFM Asia Social Network"
    The data includes the destination nodes of the provided graph.
    The total number of elements in our stream is: 27,806
    '''
    df = pd.read_csv("data/lastfm_asia_edges.csv")
    return df['node_2'].to_numpy()


if __name__ == '__main__':
    # Retrieve artificial data
    buckets = 1000
    num_hash_func = 2
    data = real_dataset()
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
    

    # Calculate the probability of overestimate items and compare it with the theoretical bound. 
    # Also, create a plot to show the observed deviation of overestimate items.
    plot_dict = {"<20": 0, "20-30": 0, "30-40": 0,
                 "40-50": 0, "50-60": 0, ">60": 0}
    overestimate_counter = 0
    for investigate_item in items:
        deviation = instance.calculate(item=investigate_item) - actual_freq_dict[investigate_item]
        if deviation >= (2 * total_elements) / buckets:
            overestimate_counter += 1
        if deviation < 20:
            plot_dict["<20"] += 1
        elif deviation >= 20 and deviation < 30:
            plot_dict["20-30"] += 1
        elif deviation >= 30 and deviation < 40:
            plot_dict["30-40"] += 1
        elif deviation >= 40 and deviation < 50:
            plot_dict["40-50"] += 1
        elif deviation >= 50 and deviation < 60:
            plot_dict["50-60"] += 1
        else:
            plot_dict[">60"] += 1

    print("Total number of elements in the stream: {}".format(len(data)))
    print("Theoretical bound of CountMinSketch with {} buckets and {} hash functions: {}".format(
        buckets, num_hash_func, 1/(pow(2, num_hash_func))))
    print("Probability of overestimating elements: {}".format(
        overestimate_counter/len(items)))

    # Creating the bar plot
    fig = plt.figure(figsize=(10, 5))
    bins = list(plot_dict.keys())
    values = list(plot_dict.values())
    y = plt.bar(bins, values, color='maroon', width=0.4)
    plt.axvline(x=4, linestyle='--')
    plt.text(4.1, 2200, '(2S)/K = ' + str((2 * total_elements) / buckets))
    plt.ylabel("# Elements")
    plt.xlabel("Deviation from Exact Value")
    plt.title("Observed Deviation Of Overestimating \n (Distinct Elements = {})".format(len(items)))
    for rect in y:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' %
                 int(height), ha='center', va='bottom', fontweight='bold')
    plt.savefig('plots/q3.png')
