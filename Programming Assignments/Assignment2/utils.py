import random
from random import shuffle
import networkx as nx
import numpy as np


class maximalMatching(object):
    def __init__(self, graph):
        self.edge_list = list(graph.edges)
        self.nodes = graph.nodes

        self.adj_matrix = self.calculate_adj_matrix()
        # Create a dictionary of
        # key: edge list index
        # value: random value between 0 and 1
        self.random_dict = {}
        for index in range(len(self.edge_list)):
            random_value = random.uniform(0, 1)
            self.random_dict.update({str(index): random_value})

        # Create maximal matching for given graph
        self.matching_list = []
        edgepoints = set()
        # Iterate over sorted dictionary by random values
        for index, _ in dict(sorted(self.random_dict.items(), key=lambda item: item[1])).items():
            edge = self.edge_list[int(index)]
            src = edge[0]
            dst = edge[1]
            if len(self.matching_list) == 0 or (src not in edgepoints and dst not in edgepoints):
                self.matching_list.append(edge)
                edgepoints.add(src)
                edgepoints.add(dst)

    def get_matching_list(self):
        return self.matching_list

    def select_random_edge(self):
        """
        :return: A uniform random edge of the given graph
        """
        random_index = random.randrange(len(self.edge_list))
        return self.edge_list[random_index]

    def calculate_adj_list(self, inspected_edge):
        """
        :param inspected_edge:  Random edge
        :return: A list of adjacency lists of inspected edge.
        """
        src = inspected_edge[0]
        dst = inspected_edge[1]
        adj_list = []
        for index, edge in enumerate(self.edge_list):
            if inspected_edge != edge and (src in edge or dst in edge):
                adj_list.append([edge, index])
        return adj_list

    def calculate_adj_matrix(self):
        """
        :return: A dictionary that includes the adjacency lists of all included graph edges.
        """
        adj_matrix = {}
        for edge in self.edge_list:
            adj_matrix.update({str(edge): self.calculate_adj_list(edge)})
        return adj_matrix

    def selected1(self, inspected_edge, count):
        """
        :param inspected_edge:  Random edge
        :param count: Number of recursive calls
        :return: selected (boolean): Points out if the inspected edge is included in maximal matching or not
                 count (int): Number of recursive calls
        """
        selected = True
        count += 1
        # Calculate adj_list
        adj_list = self.calculate_adj_list(inspected_edge)
        # Find random value of inspected edge
        for index, edge in enumerate(self.edge_list):
            if inspected_edge == edge:
                inspected_edge_index = index
        random_value_inspected_edge = self.random_dict[str(inspected_edge_index)]

        for item in adj_list:
            edge = item[0]
            index = item[1]
            random_value_edge = self.random_dict[str(index)]
            if random_value_edge < random_value_inspected_edge:
                return_selected, count = self.selected1(edge, count)
                if return_selected:
                    selected = False
        return selected, count

    def selected2(self, inspected_edge, count):
        """
        :param inspected_edge:  Random edge
        :param count: Number of recursive calls
        :return: selected (boolean): Points out if the inspected edge is included in maximal matching or not
                 count (int): Number of recursive calls
        """
        count += 1
        # Calculate adj_list
        adj_list = self.calculate_adj_list(inspected_edge)
        # Find random value of inspected edge
        for index, edge in enumerate(self.edge_list):
            if inspected_edge == edge:
                inspected_edge_index = index
        random_value_inspected_edge = self.random_dict[str(inspected_edge_index)]

        # Random permutation of adj_list
        shuffle(adj_list)

        for item in adj_list:
            edge = item[0]
            index = item[1]
            random_value_edge = self.random_dict[str(index)]
            if random_value_edge < random_value_inspected_edge:
                return_selected, count = self.selected2(edge, count)
                if return_selected:
                    return False, count
        return True, count

    def selected3(self, inspected_edge, count):
        """
        :param inspected_edge:  Random edge
        :param count: Number of recursive calls
        :return: selected (boolean): Points out if the inspected edge is included in maximal matching or not
                 count (int): Number of recursive calls
        """
        count += 1
        src = inspected_edge[0]
        dst = inspected_edge[1]
        # Calculate adj_list
        adj_list = []
        for index, edge in enumerate(self.edge_list):
            if inspected_edge != edge and (src in edge or dst in edge):
                adj_list.append([edge, self.random_dict[str(index)]])

        # Find random value of inspected edge
        for index, edge in enumerate(self.edge_list):
            if inspected_edge == edge:
                inspected_edge_index = index
        random_value_inspected_edge = self.random_dict[str(
            inspected_edge_index)]

        # Sort adj_list by random value
        adj_list.sort(key=lambda item: item[1])
        for item in adj_list:
            edge = item[0]
            random_value_edge = item[1]
            if random_value_edge < random_value_inspected_edge:
                return_selected, count = self.selected3(edge, count)
                if return_selected:
                    return False, count
        return True, count

    def selected4(self, inspected_edge, count):
        """
        My variation of exploration method to check if inspected edge is included in maximal matching or not.
        The only difference with selected2 is that the adjacency lists have already been precomputed and stored into adjacency matrix.
        :param inspected_edge:  Random edge
        :param count: Number of recursive calls
        :return: selected (boolean): Points out if the inspected edge is included in maximal matching or not
                 count (int): Number of recursive calls
        """
        selected = True
        count += 1
        # Find adj_list from precomputed adj_matrix
        adj_list = self.adj_matrix[str(inspected_edge)]
        # Find random value of inspected edge
        for index, edge in enumerate(self.edge_list):
            if inspected_edge == edge:
                inspected_edge_index = index
        random_value_inspected_edge = self.random_dict[str(
            inspected_edge_index)]

        for item in adj_list:
            edge = item[0]
            index = item[1]
            random_value_edge = self.random_dict[str(index)]
            if random_value_edge < random_value_inspected_edge:
                return_selected, count = self.selected4(edge, count)
                if return_selected:
                    selected = False
        return selected, count


class explorationMethods(object):
    def __init__(self):
        pass

    def generate_matching(self, size):
        """
        Generate a random matching of size floor(n/2).
        """
        nodes = list(range(size))
        random.shuffle(nodes)
        matching = []
        for i in range(0, size, 2):
            matching.append((nodes[i], nodes[i + 1]))
        return matching

    def create_point(self, n):
        """
        Create n random 2d points in [0,1]^2 square.
        """
        coordinates = []
        for _ in range(n):
            # random (x,y) value
            coordinates.append([random.uniform(0, 1), random.uniform(0, 1)])
        return coordinates

    def method_A(self, n, d):
        """
        Generate a graph with n vertices and approximately degree d using random matchings.
        """
        G = nx.Graph()
        for _ in range(d):
            matching = self.generate_matching(n)
            G.add_edges_from(matching)
        return G

    def method_B(self, n, d):
        """
        Generate a graph based on assigning random points in Euclidean square [0,1]^2 to each vertex v.
        For each vertex v, add edges from v to d vertices corresponding to the closest points in [0, 1]^2
        """
        G = nx.Graph()
        coordinates = self.create_point(n)
        neighborhood = {}
        for i in range(n):
            inspected_point = coordinates[i]
            euclidean_distances = {}
            for j in range(n):
                if i != j:
                    euclidean_distances.update({str(j): np.linalg.norm(
                        np.array(inspected_point) - np.array(coordinates[j]))})
            for index, _ in dict(sorted(euclidean_distances.items(), key=lambda item: item[1])).items():
                if str(i) not in neighborhood:
                    neighborhood.update({str(i): [index]})
                elif len(neighborhood[str(i)]) < d:
                    neighborhood[str(i)].append(index)
                else:
                    break
        for key, value in neighborhood.items():
            for node in value:
                G.add_edge(int(key), int(node))
        return G
