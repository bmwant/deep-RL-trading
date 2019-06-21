import os

OUTPUT_FLD = os.path.join('..', 'results')


def makedirs(fld):
    if not os.path.exists(fld):
        os.makedirs(fld)
