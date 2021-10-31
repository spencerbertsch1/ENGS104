# Spencer Bertsch
# Dartmouth College
# ENGS 104, Fall 2021
# Final Project

import random
from pathlib import Path

import click
import numpy as np

from routines import image_reader, image_to_adjacency_list, Graph

random.seed(38)

PATH_TO_THIS_FILE: Path = Path(__file__).resolve()


def driver(image_name: str, start_state, end_state):
    """
    Driver for the Shortest Path over Images project

    :param image_name: Name of the image that will be used for Dijkstras.
    :param start_state: Tuple representing the (row. col) of the start position.
    :param end_state: Tuple representing the (row. col) of the end position.
    :return:
    """

    print(f'Running Shortest Path analysis on: {image_name}')
    ABSPATH_TO_IMG: Path = PATH_TO_THIS_FILE.parent / 'images' / image_name

    # STEP 1. Read a chosen image
    image: np.ndarray = image_reader(ABSPATH_TO_IMG=ABSPATH_TO_IMG)

    # STEP 2: Convert the image to an adjacency list
    adj_list: dict = image_to_adjacency_list(img=image, distance=1, use_bresenhams=False, weight_calc='euclidean')

    # STEP 3: Runs dijkstras to find the shortest path
    g = Graph(adj_list=adj_list)
    solution = g.dijkstra(start_state=start_state, end_state=end_state)
    print(solution)

    # STEP 4: Save a png of the image with the path superimposed
    # TODO - massage the path into a list of 4-length tuples (x1, y1, x2, y2)
    # TODO - call the path plotter to see the shortest path on the image

    # STEP 5: Use simulatioin to do the following:
    # Understand the relationship between the distance value when finding the adjacency list and the path length.
    # Understand how

    # STEP 5: creates an animation of the path being generated
    # TODO

    print('something')


if __name__ == "__main__":
    driver(image_name='image1.png', start_state=(0,0), end_state=(6,6))
