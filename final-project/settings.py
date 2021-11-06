# Spencer Bertsch
# Dartmouth College
# ENGS 104, Fall 2021
# Final Project

import os
from pathlib import Path

PATH_TO_THIS_FILE: Path = Path(__file__).resolve()

ABSPATH_TO_IMAGES: Path = PATH_TO_THIS_FILE.parent / 'images'
ABSPATH_TO_SOL_IMAGES: Path = PATH_TO_THIS_FILE.parent / 'solutions'
ABSPATH_TO_SPARSE_IMAGES: Path = PATH_TO_THIS_FILE.parent / 'images' / 'sparse_imgs'

ABSPATH_TO_SPARSE_SIM_RESULTS: Path = PATH_TO_THIS_FILE.parent / 'simulation_results' / 'sparse'