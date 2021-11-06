# Spencer Bertsch
# Dartmouth College
# ENGS 104, Fall 2021
# Final Project

import os
import pickle
import collections
import random
from pathlib import Path

import matplotlib.pyplot as plt

from driver import driver
from settings import ABSPATH_TO_SPARSE_IMAGES, ABSPATH_TO_SPARSE_SIM_RESULTS


def boxplot_dict(input_dict: dict, boxplot_title: str, image_type: str = 'svg', save_results: bool = False, show_results: bool = True):
    """
    Generate a boxplot given a dictionary of keys to lists or tuples of ints or floats. The resulting image can be
    saved using different file formats such as .png or .jpeg, but .svg provides the best clarity in papers and
    presentations.

    input_dict = {
        'key1': (2.3, 3.5, 2.7, 2.8, ...),
        'key2': (6.1, 6.3, 5.8, 6.7, ...),
        ...
    }

    :param input_dict:
    :return: NA - saves and displays an image of the resulting boxplot.
    """

    ordered_dict = collections.OrderedDict(sorted(input_dict.items()))

    # here we need to turn our dict into 2 lists for plotting
    key_list = []
    value_list = []
    for key, val in ordered_dict.items():
        key_list.append(key)
        value_list.append(tuple(val))

    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle(boxplot_title)
    ax = fig.add_subplot(111)
    plt.boxplot(value_list)
    ax.set_xticklabels(key_list)

    if show_results:
        plt.show()


def plot_simulation_results(save_results: bool = False, show_results: bool = True):

    weights_path = ABSPATH_TO_SPARSE_SIM_RESULTS / 'weights.pickle'
    nodes_visited_path = ABSPATH_TO_SPARSE_SIM_RESULTS / 'nodes_visited.pickle'

    with open(str(weights_path), 'rb') as handle:
        weights_dict = pickle.load(handle)

    with open(str(nodes_visited_path), 'rb') as handle:
        nodes_visited_dict = pickle.load(handle)

    print(weights_dict, nodes_visited_dict)

    # generate a boxplot for the weights
    boxplot_dict(input_dict=weights_dict, boxplot_title='Weights', show_results=show_results,
                 save_results=save_results)

    # generate a boxplot for the nodes visited
    boxplot_dict(input_dict=nodes_visited_dict, boxplot_title='Nodes Visited', show_results=show_results,
                 save_results=save_results)

    print('Simulation results have been saves successfully')


def run_sparse_simulation():
    # run for all sparse images and retrieve the corresponding densities, nodes visited, and path weights.
    weight_dict = {}
    nodes_visited_dict = {}

    # initialize the dictionaries with empty lists for each base image
    for image_name in os.listdir(str(ABSPATH_TO_SPARSE_IMAGES)):
        if image_name.endswith(".png"):
            image_number: float = float(image_name[-9:-6])
            weight_dict[image_number] = []
            nodes_visited_dict[image_number] = []

    for image_name in os.listdir(str(ABSPATH_TO_SPARSE_IMAGES)):
        if image_name.endswith(".png"):

            # define the image number that will be the key in either of the above dictionaries
            image_number: float = float(image_name[-9:-6])

            weight, n_visited = driver(image_name=image_name,
                                       distance=1,
                                       start_state=(0, 0),
                                       end_state=(19, 19),  # <-- (29, 29), (6, 6), ...
                                       weight_calc='euclidean',  # <-- 'euclidean' or 'manhattan'
                                       four_neighbor_model=False,  # <-- 4-neighbor model or 8-neighbor model.
                                       use_bresenhams=False,
                                       create_plot=False,
                                       create_animation=False,
                                       simulation='sparse')

            # add the data from the run to the dictionaries
            weight_dict[image_number].append(weight)
            nodes_visited_dict[image_number].append(n_visited)

    # we now need to remove the solutions that were unsolvable by removing the values of 1000 from the weight_dict
    for img_density, weights in weight_dict.items():
        new_weights = [x for x in weights if x != 1000]
        weight_dict[img_density] = new_weights

    # save the results to disk so we can plot them later
    weights_path = ABSPATH_TO_SPARSE_SIM_RESULTS / 'weights.pickle'
    nodes_visited_path = ABSPATH_TO_SPARSE_SIM_RESULTS / 'nodes_visited.pickle'

    with open(str(weights_path), 'wb') as handle:
        pickle.dump(weight_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(str(nodes_visited_path), 'wb') as handle:
        pickle.dump(nodes_visited_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('successfully saved weights_dict and nodes_visited_dict.')


if __name__ == "__main__":
    # run the simulation, this step can be quite lengthy as Dijkstras is being run on many images
    run_sparse_simulation()

    # load the resulting dictionaries from disk and plot them
    plot_simulation_results()
