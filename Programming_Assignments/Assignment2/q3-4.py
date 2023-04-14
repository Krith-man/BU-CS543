import matplotlib.pyplot as plt
import utils

if __name__ == '__main__':
    results_methodA = {}
    results_methodB = {}
    num_repetitions = 100
    exploration_instance = utils.explorationMethods()
    # Iterate over number of repetitions
    for _ in range(num_repetitions):
        # Iterate over different degree number
        for d in range(2, 7):
            graph_A = exploration_instance.method_A(n=30, d=d)
            graph_Β = exploration_instance.method_B(n=30, d=d)

            # Implement "selected" functions on top of generative method(methodA, methodB)
            for graph in [graph_A, graph_Β]:
                maximal_matching_instance = utils.maximalMatching(graph)
                edge = maximal_matching_instance.select_random_edge()

                # Selected 1
                return_selected, num_recursive_calls = maximal_matching_instance.selected1(edge, -1)
                if graph == graph_A:
                    if str(d) in results_methodA:
                        results_methodA[str(d)][0] += num_recursive_calls
                    else:
                        results_methodA.update({str(d): [num_recursive_calls]})
                else:
                    if str(d) in results_methodB:
                        results_methodB[str(d)][0] += num_recursive_calls
                    else:
                        results_methodB.update({str(d): [num_recursive_calls]})

                # Selected 2
                return_selected, num_recursive_calls = maximal_matching_instance.selected2(edge, -1)
                if graph == graph_A:
                    if len(results_methodA[str(d)]) == 1:
                        results_methodA[str(d)].append(num_recursive_calls)
                    else:
                        results_methodA[str(d)][1] += num_recursive_calls
                else:
                    if len(results_methodB[str(d)]) == 1:
                        results_methodB[str(d)].append(num_recursive_calls)
                    else:
                        results_methodB[str(d)][1] += num_recursive_calls

                # Selected 3
                return_selected, num_recursive_calls = maximal_matching_instance.selected3(edge, -1)
                if graph == graph_A:
                    if len(results_methodA[str(d)]) == 2:
                        results_methodA[str(d)].append(num_recursive_calls)
                    else:
                        results_methodA[str(d)][2] += num_recursive_calls
                else:
                    if len(results_methodB[str(d)]) == 2:
                        results_methodB[str(d)].append(num_recursive_calls)
                    else:
                        results_methodB[str(d)][2] += num_recursive_calls

                # Selected 4
                return_selected, num_recursive_calls = maximal_matching_instance.selected4(edge, -1)
                if graph == graph_A:
                    if len(results_methodA[str(d)]) == 3:
                        results_methodA[str(d)].append(num_recursive_calls)
                    else:
                        results_methodA[str(d)][3] += num_recursive_calls
                else:
                    if len(results_methodB[str(d)]) == 3:
                        results_methodB[str(d)].append(num_recursive_calls)
                    else:
                        results_methodB[str(d)][3] += num_recursive_calls

    # Print results
    d_values = list(results_methodA.keys())
    for i, method in enumerate([results_methodA, results_methodB]):
        avg_selected1_values = []
        avg_selected2_values = []
        avg_selected3_values = []
        avg_selected4_values = []
        for key, value in method.items():
            avg_selected1_values.append(value[0] / num_repetitions)
            avg_selected2_values.append(value[1] / num_repetitions)
            avg_selected3_values.append(value[2] / num_repetitions)
            avg_selected4_values.append(value[3] / num_repetitions)
        avg_values = [avg_selected1_values, avg_selected2_values, avg_selected3_values, avg_selected4_values]
        for j in range(4):
            fig = plt.figure(figsize=(10, 5))
            plt.xlabel("d values", fontsize=16, fontweight='bold')
            plt.ylabel("Average repetition calls", fontsize=16, fontweight='bold')
            plt.bar(d_values, avg_values[j], color='maroon', width=0.4)
            if i == 0:
                plt.title("Selected" + str(j + 1) + " Method 2A", fontsize=16, fontweight='bold')
                plt.savefig("plots/q4_method_A/q4_selected" + str(j + 1) + ".png")
            else:
                plt.title("Selected" + str(j + 1) + " Method 2B", fontsize=16, fontweight='bold')
                plt.savefig("plots/q4_method_B/q4_selected" + str(j + 1) + ".png")
