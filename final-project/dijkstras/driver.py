
import random
from pathlib import Path

import click
import numpy as np

from routines import image_reader, image_to_adjacency_list

random.seed(38)

PATH_TO_THIS_FILE: Path = Path(__file__).resolve()


@click.command()
@click.option('--image_name', default=None, help='Name of the image that will be used for Dijkstras.')
def driver(image_name: str):

    print(f'Currently running Dijkstras on {image_name}')
    ABSPATH_TO_IMG: Path = PATH_TO_THIS_FILE.parent / 'images' / image_name

    # STEP 1. Read a chosen image
    image: np.ndarray = image_reader(ABSPATH_TO_IMG=ABSPATH_TO_IMG)

    # STEP 2: Convert the image to an adjacency list
    adj_list: dict = image_to_adjacency_list(img=image, distance=1, use_bresenhams=False)

    # STEP 3: Runs dijkstras to find the shortest path
    # TODO

    # STEP 4: Save a png of the image with the path superimposed
    # TODO - massage the path into a list of 4-length tuples (x1, y1, x2, y2)
    # TODO - call the path plotter to see the shortest path on the image

    # STEP 5: creates an animation of the path being generated
    # TODO


if __name__ == "__main__":
    driver()
