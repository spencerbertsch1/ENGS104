
from pathlib import Path
import time

import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.pyplot as cplot
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def image_reader(ABSPATH_TO_IMG: Path) -> np.ndarray:
    """
    Utility function to read one of the generated images

    :param ABSPATH_TO_IMG:
    :return:
    """
    # read the new image
    new_image = cv2.imread(str(ABSPATH_TO_IMG))

    return new_image


def image_to_adjacency_list(img: np.ndarray, distance: int, use_bresenhams: bool) -> dict:
    """
    Take a jpeg image and return an adjacency list representation of the graph that the image represents

    :param img: np.ndarray representation of loaded image
    :param distance: int representing the distance neighbors gathered in the get_neighbors function
    :return: dictionary - adjacency list for the graph
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

    return adjacency_list


def dijkstras():
    pass
    # TODO implement dijkstras for an adjacency matrix


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