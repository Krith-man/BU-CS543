import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.random_projection import johnson_lindenstrauss_min_dim
import random


def vanilla_JL_lemma(num_vectors, initial_dimensions, target_dimensions, epsilon, option, input):
    if input is None:
        # Generate a random set of data in high-dimensional space
        X = np.random.rand(num_vectors, initial_dimensions)
    else:
        # Use real high-dimensional space data
        X = input
        num_vectors = 100
    # Compute the pairwise Euclidean distances between the points
    D = squareform(pdist(X))
    # Choose a tolerance value for the relative error in the pairwise distances after projection
    if target_dimensions is None:
        # Compute the minimum number of dimensions required for the projection
        k = johnson_lindenstrauss_min_dim(n_samples=num_vectors, eps=epsilon)
    else:
        k = target_dimensions
    # Apply the projection to the data using a random projection matrix
    # The projection matrix is based on a standard normal distribution
    P = np.random.randn(k, X.shape[1]) / np.sqrt(k)
    X_proj = X @ P.T
    # Compute the pairwise Euclidean distances between the projected points
    D_proj = squareform(pdist(X_proj))
    # Compute the maximum relative error in the pairwise distances after projection
    err = np.abs(D_proj - D) / D
    err[np.isnan(err)] = 0
    if option == "max":
        return np.max(err)
    elif option == "min":
        return np.min(err)
    elif option == "mean":
        return np.mean(err)


def achlioptas_JL_lemma(num_vectors, initial_dimensions, target_dimensions, epsilon, beta, option, input, delta, case):
    if input is None:
        # Generate a random set of data in high-dimensional space
        A = np.random.rand(num_vectors, initial_dimensions)
    else:
        # Use real high-dimensional space data
        A = input
        num_vectors = 100
        initial_dimensions = 100
    # Compute the pairwise Euclidean distances between the points
    D = squareform(pdist(A))
    # Compute k0, number of dimensions required for the projection
    if target_dimensions is None:
        nominator = (4 + 2 * beta)
        denominator = (epsilon ** 2 / 2) - (epsilon ** 3 / 3)
        k0 = ((nominator / denominator) * np.log(num_vectors)).astype(np.int64)
    else:
        k0 = target_dimensions
    # Compute R random matrix
    if case == 1:
        values = np.array([-np.sqrt(3), 0, np.sqrt(3)])  # Possible values to choose from
        if delta is None:
            prob = np.array([1 / 6, 2 / 3, 1 / 6])  # Probabilities of choosing each value
        else:
            prob = np.array([np.abs(1 - delta) / 2, delta, np.abs(1 - delta) / 2])
    else:
        values = np.array([-1, 1])  # Possible values to choose from
        if delta is None:
            prob = np.array([1 / 2, 1 / 2])  # Probabilities of choosing each value
        else:
            prob = np.array([1 - delta, delta])
    R = np.random.choice(values, size=(initial_dimensions, k0), p=prob)
    # Compute E matrix as described on the paper
    E = (1 / np.sqrt(k0)) * A @ R
    # Compute the pairwise Euclidean distances between the points
    D_proj = squareform(pdist(E))
    # Compute the maximum relative error in the pairwise distances after projection
    err = np.abs(D_proj - D) / D
    err[np.isnan(err)] = 0
    if option == "max":
        return np.max(err)
    elif option == "min":
        return np.min(err)
    elif option == "mean":
        return np.mean(err)


def cohen_JL_lemma(num_vectors, initial_dimensions, target_dimensions, epsilon, option, input, case):
    if input is None:
        # Generate a random set of data in high-dimensional space
        A = np.random.rand(num_vectors, initial_dimensions)
    else:
        # Use real high-dimensional space data
        A = input
        num_vectors = 100
        initial_dimensions = 100
    # Compute the pairwise Euclidean distances between the points
    D = squareform(pdist(A))
    # Compute m, number of dimensions required for the projection
    if target_dimensions is None:
        m = (np.log(num_vectors) / (epsilon ** 2)).astype(np.int64)
    else:
        m = target_dimensions
    # randomly pick s value in range [1,m]
    s = np.random.randint(1, m + 1)
    # Compute P random matrix
    P = np.zeros(shape=(m, initial_dimensions))

    if case == 1:
        for column in range(initial_dimensions):
            # random_indexes = random.sample(range(0, m), s)
            random_indexes = np.random.choice(m, s, replace=False)
            rademachers = np.random.choice([-1, 1], size=len(random_indexes))
            for row, rademacher in zip(random_indexes, rademachers):
                P[row, column] = rademacher / np.sqrt(s)
    else:
        values = np.array([-1 / np.sqrt(s), 1 / np.sqrt(s)])  # Possible values to choose from
        prob = np.array([1 / 2, 1 / 2])  # Probabilities of choosing each value
        block_size = np.ceil(m / s)
        for column in range(initial_dimensions):
            for row in range(m):
                if row % block_size == 0:
                    if block_size > m - row:
                        random_index = np.random.randint(0, int(m - row))
                    else:
                        random_index = np.random.randint(0, int(block_size))
                    P[row + random_index, column] = np.random.choice(values, p=prob)

    # Compute output matrix as described on the paper
    output_matrix = A @ P.T
    # Compute the pairwise Euclidean distances between the points
    D_proj = squareform(pdist(output_matrix))
    # Compute the maximum relative error in the pairwise distances after projection
    err = np.abs(D_proj - D) / D
    err[np.isnan(err)] = 0
    if option == "max":
        return np.max(err)
    elif option == "min":
        return np.min(err)
    elif option == "mean":
        return np.mean(err)


def høgsgaard_JL_lemma(num_vectors, initial_dimensions, target_dimensions, epsilon, option):
    # Generate a random set of data in high-dimensional space and name it A
    A = np.random.rand(num_vectors, initial_dimensions)
    # Compute the pairwise Euclidean distances between the points
    D = squareform(pdist(A))
    # Compute m, number of dimensions required for the projection
    if target_dimensions == -1:
        m = (np.log(num_vectors) / (epsilon ** 2)).astype(np.int64)
    else:
        m = target_dimensions
    # Pick s as described on paper
    s = np.floor(np.log(num_vectors) / (epsilon * np.log(m / np.log(num_vectors))))
    # Compute P random matrix
    P = np.zeros(shape=(m, num_vectors))
    values = np.array([-1, 1])  # Possible values to choose from
    prob = np.array([1 / 2, 1 / 2])  # Probabilities of choosing each value
    for column in range(num_vectors):
        random_indexes = random.sample(range(0, m), int(s))
        for row in range(m):
            if row in random_indexes:
                P[row, column] = np.random.choice(values, p=prob) / np.sqrt(s)
    # Compute output matrix as described on the paper
    output_matrix = A @ P.T
    # Compute the pairwise Euclidean distances between the points
    D_proj = squareform(pdist(output_matrix))
    # Compute the maximum relative error in the pairwise distances after projection
    err = np.abs(D_proj - D) / D
    err[np.isnan(err)] = 0
    if option == "max":
        return np.max(err)
    elif option == "min":
        return np.min(err)
    elif option == "mean":
        return np.mean(err)
