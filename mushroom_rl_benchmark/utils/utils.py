import os, sys, ast, json
import argparse
import numpy as np
from inspect import signature
import scipy.stats as st
from tqdm import trange

def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)
    interval, _ = st.t.interval(0.95, n-1, scale=se)
    return mean, interval

def get_init_states(dataset):
    pick = True
    x_0 = list()
    for d in dataset:
        if pick:
            x_0.append(d[0])
        pick = d[-1]
    return np.array(x_0)

def be_range(n, quiet):
    if quiet:
        return range(n)
    else:
        return trange(n, leave=False)

def extract_arguments(args, method):
    intersection = lambda list1, list2: [x for x in list1 if x in list2]
    filterByKey = lambda keys, data: {x: data[x] for x in keys if x in data }
    keys = intersection(signature(method).parameters.keys(), args.keys())
    params = filterByKey(keys, args)
    return params