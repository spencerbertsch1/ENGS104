# Spencer Bertsch
# Dartmouth College
# ENGS 104, Fall 2021
# Final Project

from pathlib import Path
import time
import math
from heapq import heappush, heappop, heapify

import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.pyplot as cplot
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


class Solution:

    def __init__(self):
        self.solution_path = None
        self.solution_path_weight = 0
        self.nodes_visited = 0

    def __repr__(self):
        s = f'-------- Dijkstra solution -------- \n' \
            f'Weight of shortest path: {self.solution_path_weight} \n' \
            f'Number of vertices visited during the search: {self.nodes_visited} \n' \
            f'Shortest path from start to goal: {self.solution_path} \n'
        return s


class Node:

    def __init__(self, state: tuple, neighbors: dict):
        self.state = state
        # initialize distance to infinity
        self.distance = math.inf
        self.visited = False
        self.parent = None
        self.neighbors = neighbors

    # we need a less than method so when we heappush and heappop it will compare the distances of the nodes
    def __lt__(self, other):
        return self.distance < other.distance


class Graph:

    def __init__(self, adj_list: dict):
        # nested dictionary storing vertices, neighbors, the edge weights
        self.adj_list = adj_list
        self.num_vertices = len(list(adj_list.keys()))
        # dictionary mapping states to nodes
        self.vertex_dict = self.generate_vertex_dict()

    def generate_vertex_dict(self):
        v_dict = {}
        for state, neighbors in self.adj_list.items():
            new_node = Node(state=state, neighbors=neighbors)
            v_dict[state] = new_node
        return v_dict

    def get_neighbors(self, state):
        """
        Returns the dictionary representation of all neighbors of the state and the edge weights
        :param state: tuple of length 2 (row, col)
        :return: dict - neighbors of the state that was passed in
        """
        return self.adj_list[state]

    def dijkstra(self, start_state: tuple, end_state: tuple):
        """
        Nested dictionary adjacency list implementation of Dijkstra's algorithm.

        :param start_state:
        :param end_state:
        :return:
        """
        # we set the start vertex distance to zero
        self.vertex_dict[start_state].distance = 0

        # Put tuple pair into the priority queue
        unvisited_queue = []
        for state, node in self.vertex_dict.items():
            distance = node.distance
            d_n = (distance, node)
            unvisited_queue.append(d_n)

        # we turn the list into a head here - heappush and heappop will use the __lr__ method in Node
        heapify(unvisited_queue)

        # instantiate a solution object to update during search
        sol = Solution()

        while len(unvisited_queue) > 0:
            # Pops a node with the smallest distance
            node = heappop(unvisited_queue)
            current_node = node[1]
            # set the node to visited
            current_node.visited = True

            if current_node.state == end_state:
                # we have reached the goal!
                solution_path: list = back_chaining(search_node=current_node)
                sol.solution_path = solution_path
                # get the weight of the final step to then goal node
                weight_of_final_step = current_node.neighbors[solution_path[-2]]
                sol.solution_path_weight = round(current_node.distance + weight_of_final_step, 2)
                return sol

            for neighbor_state, weight in current_node.neighbors.items():
                sol.nodes_visited = sol.nodes_visited + 1
                # get the neighbor node from the neighbor state
                neighbor_node = self.vertex_dict[neighbor_state]

                # if we've already visited this node, then we can move on
                if neighbor_node.visited is True:
                    continue
                # calculate the new shortest distance to the node
                new_distance = current_node.distance + weight

                if new_distance < neighbor_node.distance:
                    # we have found a shorter path to the new node! So we update the distance and parent
                    neighbor_node.distance = new_distance
                    neighbor_node.parent = current_node

            # Lastly we need to rebuild the heap to reflect the changes we just made
            # remove all nodes from the heap
            while len(unvisited_queue) > 0:
                heappop(unvisited_queue)
            # add all of the nodes that have not yet been visited into the queue
            unvisited_queue = []
            for state, node in self.vertex_dict.items():
                # if we've already visited a node, then we don't add it to the queue
                if node.visited is True:
                    continue
                else:
                    distance = node.distance
                    d_n = (distance, node)
                    unvisited_queue.append(d_n)
            # heapify the list to turn it into a heap
            heapify(unvisited_queue)


def back_chaining(search_node, chain=None) -> list:
    # initialize empty path
    if chain is None:
        # add the goal state as the start of the chain
        chain = [search_node.state]

    # base case
    if search_node.parent is None:
        chain.reverse()
        return chain

    # recursive case
    chain.append(search_node.parent.state)
    return back_chaining(search_node=search_node.parent, chain=chain)


def image_reader(ABSPATH_TO_IMG: Path) -> np.ndarray:
    """
    Utility function to read one of the generated images

    :param ABSPATH_TO_IMG:
    :return:
    """
    # read the new image
    new_image = cv2.imread(str(ABSPATH_TO_IMG))

    return new_image


def image_to_adjacency_list(img: np.ndarray, distance: int, use_bresenhams: bool, weight_calc: str) -> dict:
    """
    Take a jpeg image and return an adjacency list representation of the graph that the image represents

    The nodes here are stored as nested dicts. Each node stores a state - a 2 length tuple representing
    a row/column for the given state. It maps to a dictionary of all neighbors and weights. Neighbors are
    also tuples of length 2, and the values are floats representing edge weights.

    adj_list: dict = {
        (0,0): {            <--- Node
            (0,1): 1,       <--- Neighbor and corresponding edge weight
            (1,0): 1        <--- Neighbor and corresponding edge weight
            (1,1): 1.41     <--- Neighbor and corresponding edge weight
        },
        ...
    }

    :param img: np.ndarray representation of loaded image
    :param distance: int representing the distance neighbors gathered in the get_neighbors function
    :return: dictionary - adjacency list for the graph. Nodes are dictionaries of dictionaries
    """
    print('Converting image to adjacency list')

    # initialize new list that will store lists, then later we will convert this back to np.ndarray
    new_array: list = []
    # flatten the pixels into a single number between 0 and 1
    for i in img:
        row: list = []
        for j in i:
            if j[0] != 0:
                pixel_value = 1
            else:
                pixel_value = 0
            # append the new, binary pixel value to the row list
            row.append(pixel_value)
        # append the row list to the outer new_array list
        new_array.append(row)

    # convert the new_array list of lists into an np.ndarray
    flat_img: np.ndarray = np.array(new_array)

    def find_neighbor_indices(m, i, j, dist=distance):
        # thanks to Pyrce from stackoverflow here: bit.ly/3Ek3NzO
        neighbors = []
        irange = range(max(0, i-dist), min(len(m), i+dist+1))
        if len(m) > 0:
            jrange = range(max(0, j-dist), min(len(m[0]), j+dist+1))
        else:
            jrange = []
        for icheck in irange:
            for jcheck in jrange:
                # Skip when i==icheck and j==jcheck
                if icheck != i or jcheck != j:
                    neighbors.append((icheck, jcheck))
        return neighbors

    # now it's time to create the adjacency list!
    adjacency_list = {}
    for i in range(len(new_array)):
        for j in range(len(new_array)):
            # get neighbors of current cell
            neighbors: list = find_neighbor_indices(new_array, i, j)
            valid_neighbors = []
            for n in neighbors:
                if new_array[n[0]][n[1]] == 1:
                    valid_neighbors.append(n)

            # TODO implement use_bresenhams line algorithm to discount any squares
            #  if we pass through a block to get there
            if use_bresenhams:
                # Here we would prune the cells in valid_neighbors to ONLY those achievable
                # through a straight line without hitting a blocker
                pass

            # add the cell-list pair to the adjacency list dictionary
            cell: tuple = (i, j)
            adjacency_list[cell] = valid_neighbors

    # add the path weights as either the euclidean or the manhattan distance between nodes in the graph
    new_adjacency_list = {}
    for node, neighbors in adjacency_list.items():
        neighbors_with_weights = {}
        for neighbor_node in neighbors:
            if weight_calc == 'euclidean':
                weight: float = round((math.sqrt((neighbor_node[0] - node[0])**2 + (neighbor_node[1] - node[1])**2)), 3)
            elif weight_calc == 'manhattan':
                weight: float = round(abs(neighbor_node[0] - node[0]) + abs(neighbor_node[1] - node[1]), 3)
            else:
                raise Exception(f'The \'weight_calc\' parameter should be either \'euclidean\' or '
                                f'\'manhattan\', not {weight_calc}')

            # create the k-v pair representing the neighbor node and the edge weight to that node
            neighbors_with_weights[neighbor_node] = weight

        # append the list of newly weighted nodes to the new adjacency list
        new_adjacency_list[node] = neighbors_with_weights

    # and lastly we want to remove the nodes that are'nt legal nodes
    illegal_nodes = []
    for node in new_adjacency_list.keys():
        if new_array[node[0]][node[1]] == 0:
            illegal_nodes.append(node)
    # and we can now remove the illegal nodes from the new_adjacency_list
    for node_to_remove in illegal_nodes:
        del new_adjacency_list[node_to_remove]

    return new_adjacency_list


def plot_examples(colormaps):
    """
    Helper function to plot data with associated colormap.
    """
    np.random.seed(42)
    data = np.random.randn(30, 30)
    n = len(colormaps)
    fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
                            constrained_layout=True, squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()


def animate_path(data: np.ndarray):
    processed_lines = []
    # this is what dijkstra's will return
    lines = [(0, 0, 1, 0), (1, 0, 2, 0), (2, 0, 3, 0), (3, 0, 3, 1)]
    for line in lines:
        # (x1, y1, x2, y2)
        # draw the shortest path!
        x = [line[0], line[2]]
        y = [line[1], line[3]]
        processed_lines.append((x, y))
        for line in processed_lines:
            plt.plot(line[0], line[1], color="red", linewidth=5)

        # create the image  # cmaps: 'BrBG', 'BrBG_r'
        plt.imshow(data, cmap=cplot.cm.winter, interpolation='nearest', origin='lower')
        plt.show()
        time.sleep(0.5)