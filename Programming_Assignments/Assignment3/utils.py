import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
from math import comb
import pandas as pd


def random_subroutine_1(n, size):
    """
    Generates "size" number of samples from the uniform distribution on [n].
    """
    return np.random.randint(1, n + 1, size=size)


def random_subroutine_2(n, epsilon, size):
    """
    Generates "size" number of samples from an epsilon far uniform distribution on [n].
    """
    prob_list = np.zeros(n)
    index = int(n / 2)
    prob_list[:index] = [(1 - (2 * epsilon)) / n] * index
    prob_list[index:] = [(1 + (2 * epsilon)) / n] * index
    random.shuffle(prob_list)
    return np.random.choice(np.arange(1, n + 1), size=size, p=prob_list.tolist())


def calculate_num_collisions(freq_dict):
    """
    Calculates the total number of collisions based on
    a dictionary that holds frequencies of inspected elements.
    """
    collisions = 0
    for freq_elem in freq_dict.values():
        if freq_elem > 1:
            collisions += comb(freq_elem, 2)
    return collisions


def func_2(n, epsilon, delta):
    """
    Uniformity tester of Question 2.
    :param n:  Total number of elements from which the subroutines create samples.
    :param epsilon:  Parameter of epsilon far distribution
    :param delta:  Range of probability that uniformity tester works.
    :return: Number of samples (s) and threshold (t) for which the uniformity tester holds.
    """
    s = 2500
    threshold = 335
    while True:
        num_collisions_U = []
        num_collisions_Dfar = []
        # Iterate (1 / delta) times to collect collisions over s samples
        num_iterations = int(1 / delta) * 100
        for i in range(num_iterations):
            samples_U = random_subroutine_1(n, size=s)
            samples_Dfar = random_subroutine_2(n, epsilon, size=s)
            freq_dict_U = Counter(samples_U)
            freq_dict_Dfar = Counter(samples_Dfar)
            num_collisions_U.append(calculate_num_collisions(freq_dict_U))
            num_collisions_Dfar.append(calculate_num_collisions(freq_dict_Dfar))

        # Plot the relationship between the uniform and the epsilon-far collision distributions
        _ = plt.figure()
        plt.bar(list(Counter(num_collisions_U).keys()), Counter(num_collisions_U).values(), color='g', label='U')
        plt.bar(list(Counter(num_collisions_Dfar).keys()), Counter(num_collisions_Dfar).values(), color='b', label='Dfar')
        plt.bar(threshold, max(list(Counter(num_collisions_U).values()) + list(Counter(num_collisions_Dfar).values())), color='r', label='Threshold')
        plt.title("Comparison between U and Dfar Collision Distribution \n (n=" + str(n) + ", ε=" + str(epsilon) + ", δ=" + str(delta) + ")")
        plt.xlabel("Collisions")
        plt.ylabel("Collision Frequency")
        plt.legend()
        plt.savefig("plots/q2/collision_distribution_n=" + str(n) + ".png")

        # Calculate the threshold
        below_threshold_fraction = sum(np.array(num_collisions_U) < threshold) / len(num_collisions_U)
        print("Below threshold fraction: {} %".format(below_threshold_fraction * 100))
        above_threshold_fraction = sum(np.array(num_collisions_Dfar) > threshold) / len(num_collisions_Dfar)
        print("Above threshold fraction: {} %".format(above_threshold_fraction * 100))
        if below_threshold_fraction >= (1 - delta) and above_threshold_fraction >= (1 - delta):
            print("Samples: {} Threshold: {}".format(s, threshold))
            return True
        # else:
        #     s = int(s * 1.1)


def uniformity_test_4(delta, case):
    """
    Uniformity tester of Question 4.
    :param delta:  Range of probability that uniformity tester works.
    :param case: Either use the function for "single digit" or "pair" scenarios.
    :return: Checks if the real dataset samples follow a uniform distribution or not.
    """
    # Extract data from real world dataset
    filename = 'data/exchange_dataset.csv'
    df = pd.read_csv(filename)
    second_decimal = np.array((df['open'] * 100) % 10).astype(int)
    third_decimal = np.array((df['open'] * 1000) % 10).astype(int)
    if case == "single digit":
        print("Checking real dataset with single digit samples")
        s = 102
        threshold = 545
        n = 10
    elif case == "pairs":
        print("Checking real dataset with pair of digits samples")
        s = 260
        threshold = 360
        n = 100
        pairs = []
        for i in range(len(second_decimal)):
            pairs.append(int(str(second_decimal[i]) + str(third_decimal[i])))
    # Iterate more than (1 / delta) times to collect collisions over s samples
    num_iterations = int(1 / delta) * 100
    num_collisions_U = []
    num_collisions_dataset = []
    for i in range(num_iterations):
        samples_U = random_subroutine_1(n, size=s)
        if case == "single digit":
            random.shuffle(second_decimal)
            samples_dataset = second_decimal[:s]
        elif case == "pairs":
            random.shuffle(pairs)
            samples_dataset = pairs[:s]
        freq_dict_U = Counter(samples_U)
        freq_dict_dataset = Counter(samples_dataset)
        num_collisions_U.append(calculate_num_collisions(freq_dict_U))
        num_collisions_dataset.append(calculate_num_collisions(freq_dict_dataset))

    # Plot the relationship between the uniform and the epsilon-far collision distributions
    _ = plt.figure()
    plt.bar(list(Counter(num_collisions_U).keys()), Counter(num_collisions_U).values(), color='g', label='U')
    plt.bar(list(Counter(num_collisions_dataset).keys()), Counter(num_collisions_dataset).values(), color='b', label='Real Dataset')
    plt.bar(threshold, max(list(Counter(num_collisions_U).values()) + list(Counter(num_collisions_dataset).values())), color='r', label='Threshold')
    plt.title("Comparison between U and Real Dataset Collision Distribution \n (n=" + str(n) + ", ε=" + str(0.2) + ", δ=" + str(delta) + ")")
    plt.xlabel("Collisions")
    plt.ylabel("Collision Frequency")
    plt.legend()
    plt.savefig("plots/q4/dataset_n=" + str(n) + ".png")

    # Calculate the threshold
    above_threshold_fraction = sum(np.array(num_collisions_dataset) > threshold) / len(num_collisions_dataset)
    print("Above threshold fraction: {} %".format(above_threshold_fraction * 100))
    if above_threshold_fraction >= (1 - delta):
        print("Dataset's distribution is not uniform")
    else:
        print("Dataset's distribution is uniform")


def create_sequence_digits(q, num_seq):
    """
    Creates a sequence of digits and writes them in a .txt file for later usage.
    :param q:  Number of randomly created digits
    :param num_seq:  Total number of output random digits written to txt file
    """
    random_digits = []
    for _ in range(num_seq):
        random_digits.append(np.random.randint(0, 10, size=q).sum() % 10)
    df = pd.DataFrame(random_digits, columns=['digits'])
    filename = 'data/digits_num_seq=' + str(num_seq) + '_q=' + str(q) + '.csv'
    df.to_csv(filename, index=False)


def uniformity_test_5(q, delta, case):
    """
    Uniformity tester of Question 5.
    :param q: parameter which generates input digits as described in exercise.
    :param delta:  Range of probability that uniformity tester works.
    :param case: Either use the function for "single digit" or "pair" or "triples" scenarios.
    :return: Checks if the inspected samples follow a uniform distribution or not.
    """
    # Extract data from txt file
    num_seq = 10000
    filename = 'data/digits_num_seq=' + str(num_seq) + '_q=' + str(q) + '.csv'
    df = pd.read_csv(filename)
    random_digits = df['digits'].values.tolist()
    if case == "single digit":
        print("Checking with single digit samples")
        s = 405
        threshold = 8291
        epsilon = 0.1
        n = 10
    elif case == "pairs":
        print("Checking with pair of digits samples")
        s = 260
        threshold = 360
        epsilon = 0.2
        n = 100
    elif case == "triples":
        print("Checking with triples of digits samples")
        s = 800
        threshold = 345
        epsilon = 0.2
        n = 1000
    # Iterate more than (1 / delta) times to collect collisions over s samples
    num_iterations = int(1 / delta) * 100
    num_collisions_U = []
    num_collisions_q5 = []
    samples_q5 = []
    for i in range(num_iterations):
        samples_U = random_subroutine_1(n, size=s)
        if case == "single digit":
            random.shuffle(random_digits)
            samples_q5 = random_digits[:s]
        elif case == "pairs":
            random.shuffle(random_digits)
            samples_q5 = []
            for i in range(0, 2 * s, 2):
                samples_q5.append(int(str(random_digits[i]) + str(random_digits[i + 1])))
        elif case == "triples":
            random.shuffle(random_digits)
            samples_q5 = []
            for i in range(0, 3 * s, 3):
                samples_q5.append(int(str(random_digits[i]) + str(random_digits[i + 1]) + str(random_digits[i + 2])))
        freq_dict_U = Counter(samples_U)
        freq_dict_q5 = Counter(samples_q5)
        num_collisions_U.append(calculate_num_collisions(freq_dict_U))
        num_collisions_q5.append(calculate_num_collisions(freq_dict_q5))

    # Plot the relationship between the uniform and the epsilon-far collision distributions
    _ = plt.figure()
    plt.bar(list(Counter(num_collisions_U).keys()), Counter(num_collisions_U).values(), color='g', label='U')
    plt.bar(list(Counter(num_collisions_q5).keys()), Counter(num_collisions_q5).values(), color='b', label='Question 5 Distribution')
    plt.bar(threshold, max(list(Counter(num_collisions_U).values()) + list(Counter(num_collisions_q5).values())), color='r', label='Threshold')
    plt.title("Comparison between U and Question 5 Collision Distribution \n (n=" + str(n) + ", ε=" + str(epsilon) + ", δ=" + str(delta) + ")")
    plt.xlabel("Collisions")
    plt.ylabel("Collision Frequency")
    plt.legend()
    if case == "single digit":
        plt.savefig("plots/q5/sd_q=" + str(q) + ".png")
    elif case == "pairs":
        plt.savefig("plots/q5/p_q=" + str(q) + ".png")
    elif case == "triples":
        plt.savefig("plots/q5/t_q=" + str(q) + ".png")
    # Calculate the threshold
    above_threshold_fraction = sum(np.array(num_collisions_q5) > threshold) / len(num_collisions_q5)
    print("Above threshold fraction: {} %".format(above_threshold_fraction * 100))
    if above_threshold_fraction >= (1 - delta):
        print("Question 5 distribution is not uniform")
    else:
        print("Question 5 distribution is uniform")
