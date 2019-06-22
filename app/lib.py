import os

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir))
OUTPUT_FLD = os.path.join(ROOT_DIR, 'results')
# length of dataset used when training
DATASET_LENGTH = 180  # default to 6 months


def makedirs(fld):
    if not os.path.exists(fld):
        os.makedirs(fld)
