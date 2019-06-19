import random, os, datetime, pickle, json, keras, sys
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np

OUTPUT_FLD = os.path.join('..', 'results')


def makedirs(fld):
    if not os.path.exists(fld):
        os.makedirs(fld)
