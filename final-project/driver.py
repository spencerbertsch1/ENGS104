# Spencer Bertsch
# Dartmouth College
# ENGS 104, Fall 2021
# Final Project

import random
from pathlib import Path

import click
import numpy as np

from routines import image_reader, image_to_adjacency_list, Graph, animate_path, save_solution_img
from settings import ABSPATH_TO_IMAGES, ABSPATH_TO_SOL_IMAGES, ABSPATH_TO_SPARSE_IMAGES
random.seed(38)


def driver(image_name: str, start_state: tuple, end_state: tuple, weight_calc: str,
           create_plot: bool, create_animation: bool, distance: int, use_bresenhams: bool,
           four_neighbor_model: bool, simulation: str):
    """
    Driver for the Shortest Path over Images project

    :param image_name: Name of the image that will be used for Dijkstras.
    :param start_state: Tuple representing the (row. col) of the start position.
    :param end_state: Tuple representing the (row. col) of the end position.
    :return:
    """

    print(f'Running Shortest Path analysis on: {image_name}')
    if simulation == 'sparse':
        ABSPATH_TO_IMG: Path = ABSPATH_TO_SPARSE_IMAGES / image_name
    else:
        ABSPATH_TO_IMG: Path = ABSPATH_TO_IMAGES / image_name
    ABSPATH_TO_SOL_IMG: Path = ABSPATH_TO_SOL_IMAGES / image_name

    # STEP 1. Read a chosen image
    image: np.ndarray = image_reader(ABSPATH_TO_IMG=ABSPATH_TO_IMG)

    # STEP 2: Convert the image to an adjacency list
    adj_list: dict = image_to_adjacency_list(img=image, distance=distance, use_bresenhams=False,
                                             weight_calc=weight_calc, four_neighbor_model=four_neighbor_model)

    # STEP 3: Runs dijkstras to find the shortest path
    g = Graph(adj_list=adj_list)
    solution = g.dijkstra(start_state=start_state, end_state=end_state)
    print(solution)

    # STEP 4: Save a png of the image with the path superimposed
    if create_plot:
        save_solution_img(np_image=image, fpath=str(ABSPATH_TO_SOL_IMG), chain=solution.solution_path, show_img=True,
                          weight=solution.solution_path_weight, neighbor_distance=distance,
                          use_bresenhams=use_bresenhams)

    # STEP 5: creates an animation of the path being generated
    if create_animation:
        animate_path(np_image=image, fpath=str(ABSPATH_TO_SOL_IMG), chain=solution.solution_path, save_final_img=True)

    # STEP 6: Use simulation to do the following:
    # Understand the relationship between the distance value when finding the adjacency list and the path length.
    # Understand how

    return solution.solution_path_weight, solution.nodes_visited


if __name__ == "__main__":
    driver(image_name='image30.png',
           distance=1,
           start_state=(0, 0),
           end_state=(29, 29),  # <-- (29, 29), (6, 6), ...
           weight_calc='euclidean',  # <-- 'euclidean' or 'manhattan'
           four_neighbor_model=False,  # <-- 4-neighbor model or 8-neighbor model. Only use 4-neighbor when distance=1
           use_bresenhams=True,
           create_plot=True,
           create_animation=False,
           simulation='none')
