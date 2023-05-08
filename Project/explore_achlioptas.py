import numpy as np
import matplotlib.pyplot as plt
import utils


def explore_achlioptas(option, case, num_repetitions):
    np.seterr(invalid='ignore')
    _ = plt.figure(figsize=(10, 6))
    epsilon = 0.1
    deltas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    distortion = []
    plt.xlabel('Delta')
    for delta in deltas:
        result = 0
        for _ in range(num_repetitions):
            result += utils.achlioptas_JL_lemma(num_vectors=100, initial_dimensions=10000, target_dimensions=2000,
                                                epsilon=epsilon, beta=0.1, option=option, input=None, delta=delta,
                                                case=case)
        distortion.append(result / num_repetitions)

    plt.title('Achlioptas JL lemma: Sparsity Experiment with Target Dimensions=2000')
    if option == "max":
        plt.ylabel('Maximum Distortion')
        plt.plot(deltas, distortion)
        if case == 1:
            plt.savefig("figures/experiment_explore_achlioptas/max/achlioptas_case_1_deltas.png")
        else:
            plt.savefig("figures/experiment_explore_achlioptas/max/achlioptas_case_0_deltas.png")
    elif option == "min":
        plt.ylabel('Minimum Distortion')
        plt.plot(deltas, distortion)
        if case == 1:
            plt.savefig("figures/experiment_explore_achlioptas/min/achlioptas_case_1_deltas.png")
        else:
            plt.savefig("figures/experiment_explore_achlioptas/min/achlioptas_case_0_deltas.png")
    elif option == "mean":
        plt.ylabel('Mean Distortion')
        plt.plot(deltas, distortion)
        if case == 1:
            plt.savefig("figures/experiment_explore_achlioptas/mean/achlioptas_case_1_deltas.png")
        else:
            plt.savefig("figures/experiment_explore_achlioptas/mean/achlioptas_case_0_deltas.png")


if __name__ == '__main__':
    # Explore Achlioptas matrix sparsity
    explore_achlioptas(option="max", case=1, num_repetitions=10)
    explore_achlioptas(option="min", case=1, num_repetitions=10)
    explore_achlioptas(option="mean", case=1, num_repetitions=10)
    explore_achlioptas(option="max", case=0, num_repetitions=10)
    explore_achlioptas(option="min", case=0, num_repetitions=10)
    explore_achlioptas(option="mean", case=0, num_repetitions=10)
