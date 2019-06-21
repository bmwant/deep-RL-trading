import os

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir))
OUTPUT_FLD = os.path.join(ROOT_DIR, 'results')


def makedirs(fld):
    if not os.path.exists(fld):
        os.makedirs(fld)
