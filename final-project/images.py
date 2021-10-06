import numpy as np
from matplotlib import pyplot as plt
import random
import time
import matplotlib.pyplot as cplot
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
random.seed(38)

# TODO break the code up into an image generator and a path plotter
# pass the image matrix to dijkstras to get the shortest path
# massage the path into a list of 4-length tuples (x1, y1, x2, y2)
# call the path plotter to see the shortest path on the image

def build_img():
    img_size: int = 30
    land_density: float = 0.95  # closer to 1 will be more water

    data = np.zeros((img_size, img_size, 1), dtype=np.uint8)

    land_squares = round(img_size**2 * land_density)
    for i in range(land_squares):
        # replace water with land
        data[random.randrange(img_size), random.randrange(img_size)] = [255]

    # ensure start and end nodes are land
    data[0, 0] = [255]
    data[img_size-1, img_size-1] = [255]

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


if __name__ == "__main__":
    cmap = ListedColormap(["darkorange", "gold", "lawngreen", "lightseagreen"])
    plot_examples([cmap])