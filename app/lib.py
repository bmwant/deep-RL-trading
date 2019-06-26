import os


ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
OUTPUT_FLD = os.path.join(ROOT_DIR, 'results')
# length of dataset used when training
DATASET_LENGTH = 180  # default to 6 months
DEFAULT_MA_WINDOW = 100
MAX_TRANSACTIONS = 10

ACTIONS = {
    0: 'sell',
    1: 'buy',
    2: 'idle',
}


def makedirs(fld):
    if not os.path.exists(fld):
        os.makedirs(fld)
