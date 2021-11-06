# Spencer Bertsch
# Dartmouth College
# ENGS 104, Fall 2021
# Final Project

import numpy as np
import random
from pathlib import Path
from PIL import Image
import cv2

PATH_TO_THIS_FILE: Path = Path(__file__).resolve()


def build_img(img_size: int, land_density: float, path_to_image: str):
    """
    Generates a random, square image of land and water, then saves the resulting image to disk.

    :param img_size: int - width/height for the generated image map
    :param land_density: (float between 0 and 1) density of water in image; closer to 1 will be more water
    :param img_size: string representing the absolute path to the saved image
    :return:
    """
    data = np.zeros((img_size, img_size, 1), dtype=np.uint8)

    land_squares = round(img_size**2 * land_density)
    for i in range(land_squares):
        # replace water with land
        data[random.randrange(img_size), random.randrange(img_size)] = [255]

    # ensure corners are land so we can start or end in any corner later on
    data[0, 0] = [255]
    data[img_size-1, img_size-1] = [255]
    data[0, img_size-1] = [255]
    data[img_size-1, 0] = [255]

    # use cvs to create a jpeg that we can use later with dijkstras
    info = np.iinfo(data.dtype)  # Get the information of the incoming image type
    data = data.astype(np.float64) / info.max  # normalize the data to 0 - 1
    data = 255 * data  # Now scale by 255

    # TODO replace all [255,255,255] pixels with [102, 51, 0] pixels and replace the water as well

    img = data.astype(np.uint8)
    cv2.imwrite(path_to_image, img)
    print(f'Image successfully saved at the following location: \n {path_to_image}')


def generate_sparse_images(side_length: int):
    """
    Generates many sparse images with varying densities for simulation testing

    :param side_length: number of pixels per side for the square images created
    :return:
    """
    num_images_per_density = 9

    for i in np.arange(3, 1.0, -0.1):
        i = round(i, 2)
        for j in range(num_images_per_density):
            j += 1
            file_name: str = f"image{side_length}_density{i}_{j}.png"
            ABSPATH_TO_IMG: Path = PATH_TO_THIS_FILE.parent / 'images' / 'sparse_imgs' / file_name
            build_img(img_size=side_length, land_density=i, path_to_image=str(ABSPATH_TO_IMG))


if __name__ == "__main__":

    # we can change the random seed to alter how the image looks
    random.seed(2)
    side_length: int = 20

    # generate the sparse imaged we will use for testing
    generate_sparse_images(side_length=side_length)
