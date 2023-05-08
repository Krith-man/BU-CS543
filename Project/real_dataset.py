import numpy as np
from sklearn.random_projection import johnson_lindenstrauss_min_dim
import matplotlib.pyplot as plt
import pandas as pd
import os
import utils


def run_experiment(JL_lemma, option, input, case, num_repetitions):
    np.seterr(invalid='ignore')
    _ = plt.figure(figsize=(10, 6))
    epsilon = 0.1
    if input is None:
        targets = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    else:
        targets = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    distortion = []
    plt.xlabel('Target Dimensions')
    if JL_lemma == "vanilla":
        for target in targets:
            print("Targets: {}".format(target))
            result = 0
            for _ in range(num_repetitions):
                result += utils.vanilla_JL_lemma(num_vectors=100, initial_dimensions=10000, target_dimensions=target, epsilon=epsilon, option=option, input=input)
            distortion.append(result / num_repetitions)
        expected_target_dimensions = johnson_lindenstrauss_min_dim(n_samples=100, eps=epsilon)
        plt.title('Vanilla Johnson-Lindenstrauss lemma')
        if option == "max":
            plt.ylabel('Maximum Distortion')
            plt.plot(targets, distortion)
            plt.axvline(x=expected_target_dimensions, color='g', linestyle='--', label='Theoretical Target Dimension')
            plt.axhline(y=epsilon, color='r', linestyle='--', label='Threshold Distortion')
            plt.legend()
            if input is None:
                plt.savefig("figures/experiment_synthetic_dataset/max/vanilla.png")
            else:
                plt.savefig("figures/experiment_real_dataset/max/vanilla.png")
        elif option == "min":
            plt.ylabel('Minimum Distortion')
            plt.plot(targets, distortion)
            plt.axvline(x=expected_target_dimensions, color='g', linestyle='--', label='Theoretical Target Dimension')
            plt.axhline(y=epsilon, color='r', linestyle='--', label='Threshold Distortion')
            plt.legend()
            if input is None:
                plt.savefig("figures/experiment_synthetic_dataset/min/vanilla.png")
            else:
                plt.savefig("figures/experiment_real_dataset/min/vanilla.png")
        elif option == "mean":
            plt.ylabel('Mean Distortion')
            plt.plot(targets, distortion)
            plt.axvline(x=expected_target_dimensions, color='g', linestyle='--', label='Theoretical Target Dimension')
            plt.axhline(y=epsilon, color='r', linestyle='--', label='Threshold Distortion')
            plt.legend()
            if input is None:
                plt.savefig("figures/experiment_synthetic_dataset/mean/vanilla.png")
            else:
                plt.savefig("figures/experiment_real_dataset/mean/vanilla.png")

    elif JL_lemma == "achlioptas":
        beta = 0.1
        for target in targets:
            result = 0
            for _ in range(num_repetitions):
                result += utils.achlioptas_JL_lemma(num_vectors=100, initial_dimensions=10000, target_dimensions=target,
                                                    epsilon=epsilon, beta=beta, option=option, input=input, delta=None, case=case)
            distortion.append(result / num_repetitions)
            nominator = (4 + 2 * beta)
            denominator = (epsilon ** 2 / 2) - (epsilon ** 3 / 3)
            expected_target_dimensions = ((nominator / denominator) * np.log(100)).astype(np.int64)
        plt.title('Achlioptas Johnson-Lindenstrauss lemma')
        if option == "max":
            plt.ylabel('Maximum Distortion')
            plt.plot(targets, distortion)
            plt.axvline(x=expected_target_dimensions, color='g', linestyle='--', label='Theoretical Target Dimension')
            plt.axhline(y=epsilon, color='r', linestyle='--', label='Threshold Distortion')
            plt.legend()
            if input is None:
                if case == 1:
                    plt.savefig("figures/experiment_synthetic_dataset/max/achlioptas_case_1.png")
                else:
                    plt.savefig("figures/experiment_synthetic_dataset/max/achlioptas_case_0.png")
            else:
                if case == 1:
                    plt.savefig("figures/experiment_real_dataset/max/achlioptas_case_1.png")
                else:
                    plt.savefig("figures/experiment_real_dataset/max/achlioptas_case_0.png")
        elif option == "min":
            plt.ylabel('Minimum Distortion')
            plt.plot(targets, distortion)
            plt.axvline(x=expected_target_dimensions, color='g', linestyle='--', label='Theoretical Target Dimension')
            plt.axhline(y=epsilon, color='r', linestyle='--', label='Threshold Distortion')
            plt.legend()
            if input is None:
                if case == 1:
                    plt.savefig("figures/experiment_synthetic_dataset/min/achlioptas_case_1.png")
                else:
                    plt.savefig("figures/experiment_synthetic_dataset/min/achlioptas_case_0.png")
            else:
                if case == 1:
                    plt.savefig("figures/experiment_real_dataset/min/achlioptas_case_1.png")
                else:
                    plt.savefig("figures/experiment_real_dataset/min/achlioptas_case_0.png")
        elif option == "mean":
            plt.ylabel('Mean Distortion')
            plt.plot(targets, distortion)
            plt.axvline(x=expected_target_dimensions, color='g', linestyle='--', label='Theoretical Target Dimension')
            plt.axhline(y=epsilon, color='r', linestyle='--', label='Threshold Distortion')
            plt.legend()
            if input is None:
                if case == 1:
                    plt.savefig("figures/experiment_synthetic_dataset/mean/achlioptas_case_1.png")
                else:
                    plt.savefig("figures/experiment_synthetic_dataset/mean/achlioptas_case_0.png")
            else:
                if case == 1:
                    plt.savefig("figures/experiment_real_dataset/mean/achlioptas_case_1.png")
                else:
                    plt.savefig("figures/experiment_real_dataset/mean/achlioptas_case_0.png")
    elif JL_lemma == "cohen":
        for target in targets:
            result = 0
            for _ in range(num_repetitions):
                result += utils.cohen_JL_lemma(num_vectors=100, initial_dimensions=10000, target_dimensions=target, epsilon=epsilon, option=option, input=input, case=case)
                print("Target: {}".format(target))
            distortion.append(result / num_repetitions)
            expected_target_dimensions = johnson_lindenstrauss_min_dim(n_samples=100, eps=epsilon)
        plt.title('Cohen Johnson-Lindenstrauss lemma')
        if option == "max":
            plt.ylabel('Maximum Distortion')
            plt.plot(targets, distortion)
            plt.axvline(x=expected_target_dimensions, color='g', linestyle='--', label='Theoretical Target Dimension')
            plt.axhline(y=epsilon, color='r', linestyle='--', label='Threshold Distortion')
            plt.legend()
            if input is None:
                if case == 1:
                    plt.savefig("figures/experiment_synthetic_dataset/max/cohen_case_1.png")
                else:
                    plt.savefig("figures/experiment_synthetic_dataset/max/cohen_case_0.png")
            else:
                if case == 1:
                    plt.savefig("figures/experiment_real_dataset/max/cohen_case_1.png")
                else:
                    plt.savefig("figures/experiment_real_dataset/max/cohen_case_0.png")
        elif option == "min":
            plt.ylabel('Minimum Distortion')
            plt.plot(targets, distortion)
            plt.axvline(x=expected_target_dimensions, color='g', linestyle='--', label='Theoretical Target Dimension')
            plt.axhline(y=epsilon, color='r', linestyle='--', label='Threshold Distortion')
            plt.legend()
            if input is None:
                if case == 1:
                    plt.savefig("figures/experiment_synthetic_dataset/min/cohen_case_1.png")
                else:
                    plt.savefig("figures/experiment_synthetic_dataset/min/cohen_case_0.png")
            else:
                if case == 1:
                    plt.savefig("figures/experiment_real_dataset/min/cohen_case_1.png")
                else:
                    plt.savefig("figures/experiment_real_dataset/min/cohen_case_0.png")
        elif option == "mean":
            plt.ylabel('Mean Distortion')
            plt.plot(targets, distortion)
            plt.axvline(x=expected_target_dimensions, color='g', linestyle='--', label='Theoretical Target Dimension')
            plt.axhline(y=epsilon, color='r', linestyle='--', label='Threshold Distortion')
            plt.legend()
            if input is None:
                if case == 1:
                    plt.savefig("figures/experiment_synthetic_dataset/mean/cohen_case_1.png")
                else:
                    plt.savefig("figures/experiment_synthetic_dataset/mean/cohen_case_0.png")
            else:
                if case == 1:
                    plt.savefig("figures/experiment_real_dataset/mean/cohen_case_1.png")
                else:
                    plt.savefig("figures/experiment_real_dataset/mean/cohen_case_0.png")


if __name__ == '__main__':
    # Real dataset experiments
    current_path = os.getcwd()
    features = pd.read_csv(current_path + '/products/raw/node-feat.csv.gz', compression='gzip', header=None).values
    # Max distortion
    run_experiment("vanilla", option="max", input=features[:100], case=None, num_repetitions=10)
    run_experiment("achlioptas", option="max", input=features[:100], case=0, num_repetitions=10)
    run_experiment("achlioptas", option="max", input=features[:100], case=1, num_repetitions=10)
    run_experiment("cohen", option="max", input=features[:100], case=0, num_repetitions=10)
    run_experiment("cohen", option="max", input=features[:100], case=1, num_repetitions=10)
    # Min distortion
    run_experiment("vanilla", option="min", input=features[:100], case=None, num_repetitions=10)
    run_experiment("achlioptas", option="min", input=features[:100], case=0, num_repetitions=10)
    run_experiment("achlioptas", option="min", input=features[:100], case=1, num_repetitions=10)
    run_experiment("cohen", option="min", input=features[:100], case=0, num_repetitions=10)
    run_experiment("cohen", option="min", input=features[:100], case=1, num_repetitions=10)
    # Mean distortion
    run_experiment("vanilla", option="mean", input=features[:100], case=None, num_repetitions=10)
    run_experiment("achlioptas", option="mean", input=features[:100], case=0, num_repetitions=10)
    run_experiment("achlioptas", option="mean", input=features[:100], case=1, num_repetitions=10)
    run_experiment("cohen", option="mean", input=features[:100], case=0, num_repetitions=10)
    run_experiment("cohen", option="mean", input=features[:100], case=1, num_repetitions=10)
