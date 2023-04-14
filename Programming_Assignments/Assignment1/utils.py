import numpy as np
import sympy

class CountMinSketch(object):
    def __init__(self, buckets, num_hash_func, max_value):
        ''' 
        Initialize the data structure
        @param buckets       (int): Number of columns in countMin sketch table
        @param num_hash_func (int): Number of rows in countMin sketch table
        @param max_value     (int): Maximum number of input stream
        '''
        self.buckets = buckets
        self.num_hash_func = num_hash_func
        self.space = np.zeros([num_hash_func, buckets])

        # Create a,b and p (prime) parameters 
        # for universal hash function family: (a * x + b) % p
        self.prime = sympy.nextprime(max_value)
        random_values = np.random.randint(
            low=1, high=self.prime, size=2 * num_hash_func)
        self.a = random_values[:num_hash_func]
        self.b = random_values[num_hash_func:]

    def universal_hash_func(self, item, i):
        ''' 
        Returns the output of i-th universal hash function for given item
        @param item (int): Input number
        @param i    (int): Hash index number 
        '''
        return (self.a[i] * item + self.b[i]) % self.prime

    def addItem(self, item):
        ''' 
        Add the given item to the countMin sketch table
        @param item (int): Input number
        '''
        for i in range(self.num_hash_func):
            index = self.universal_hash_func(item, i) % self.buckets
            self.space[i, index] += 1

    def calculate(self, item):
        ''' 
        Calculates the frequency of the given item based on the countMin sketch table
        @param item (int): Input number
        '''
        # Calculate the minimum value of frequency 
        # for a given item based on countMin sketch table
        for i in range(self.num_hash_func):
            index = self.universal_hash_func(item, i) % self.buckets
            if i == 0:
                min_value = self.space[i, index]
            else:
                if self.space[i, index] < min_value:
                    min_value = self.space[i, index]
        return min_value