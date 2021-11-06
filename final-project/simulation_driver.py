# Spencer Bertsch
# Dartmouth College
# ENGS 104, Fall 2021
# Final Project

import os
import random
from pathlib import Path

import pandas as pd

from driver import driver
from settings import ABSPATH_TO_SPARSE_IMAGES, ABSPATH_TO_SPARSE_SIM_RESULTS


def run_sparse_simulation():
    # run for all sparse images and retrieve the corresponding densities, nodes visited, and path weights.
    rows = []
    for image_name in os.listdir(str(ABSPATH_TO_SPARSE_IMAGES)):
        if image_name.endswith(".png"):

            image_number: float = float(image_name[-7:-4])

            weight, n_visited = driver(image_name=image_name,
                                       distance=1,
                                       start_state=(0, 0),
                                       end_state=(29, 29),  # <-- (29, 29), (6, 6), ...
                                       weight_calc='euclidean',  # <-- 'euclidean' or 'manhattan'
                                       four_neighbor_model=False,  # <-- 4-neighbor model or 8-neighbor model.
                                       use_bresenhams=False,
                                       create_plot=True,
                                       create_animation=False,
                                       simulation='sparse')

            p = pd.DataFrame([image_number, n_visited, weight])
            rows.append(p)

    # why are all the path weights 59?...
    df = pd.concat(rows, axis=1).T
    df.sort_values(by=[0], inplace=True)
    df.index = df[0]
    df = df[[1, 2]]
    df.columns = ['nodes_visited', 'sp_weight']

    results_path = ABSPATH_TO_SPARSE_SIM_RESULTS / 'results.csv'
    df.to_csv(results_path, sep=',')


if __name__ == "__main__":
    run_sparse_simulation()
