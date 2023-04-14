import utils
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Create a graph from txt file
    G = nx.Graph()
    path = 'data/graph.txt'
    with open(path, 'r') as file:
        for line in file:
            edge = line.strip().split(' ')
            src = int(edge[0])
            dst = int(edge[1])
            G.add_edge(src, dst)
    # Save graph figure
    nx.draw(G, with_labels=True)
    plt.savefig('plots/q1.png')

    # Implement "selected" functions on top of generated graph
    maximal_matching_instance = utils.maximalMatching(G)
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
