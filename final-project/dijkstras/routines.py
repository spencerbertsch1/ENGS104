
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


def image_to_adjacency_matrix(img: np.ndarray) -> np.ndarray:
    print('Converting image to adjacency matrix')

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

    print('something')

    return flat_img


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