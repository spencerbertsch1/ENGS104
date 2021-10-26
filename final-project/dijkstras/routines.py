
from pathlib import Path
import cv2


def image_reader(ABSPATH_TO_IMG: Path):
    # read the new image
    new_image = cv2.imread(str(ABSPATH_TO_IMG))

    return new_image


def image_to_adjacency_matrix():
    pass
    # TODO write a function that converts an rgb image into an adjacency matrix


def dijkstras():
    pass
    # TODO implement dijkstras for an adjacency matrix
