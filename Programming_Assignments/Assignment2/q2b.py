import networkx as nx
import matplotlib.pyplot as plt
import utils

if __name__ == '__main__':
    # Create a graph with given number of nodes (n) with a roughly degree (d)
    exploration_instance = utils.explorationMethods()
    graph = exploration_instance.method_B(n=6, d=2)
    # Save graph figure
    nx.draw(graph, with_labels=True)
    plt.savefig('plots/q2b.png')

    # Implement "selected" functions on top of generated graph
    maximal_matching_instance = utils.maximalMatching(graph)
    edge = maximal_matching_instance.select_random_edge()
    print("Maximal Matching list {}".format(maximal_matching_instance.get_matching_list()))
    print("Random edge: {}".format(edge))

    return_selected, num_recursive_calls = maximal_matching_instance.selected1(edge, -1)
    print("Output selected1 function: {}. Number of recursive calls {}".format(
        return_selected, num_recursive_calls))
    return_selected, num_recursive_calls = maximal_matching_instance.selected2(edge, -1)
    print("Output selected2 function: {}. Number of recursive calls {}".format(
        return_selected, num_recursive_calls))
    return_selected, num_recursive_calls = maximal_matching_instance.selected3(edge, -1)
    print("Output selected3 function: {}. Number of recursive calls {}".format(
        return_selected, num_recursive_calls))
    return_selected, num_recursive_calls = maximal_matching_instance.selected4(edge, -1)
    print("Output selected4 function: {}. Number of recursive calls {}".format(
        return_selected, num_recursive_calls))
