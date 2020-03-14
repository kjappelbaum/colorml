# -*- coding: utf-8 -*-
"""
Filename: /home/kevin/Dropbox (LSMO)/proj75_mofcolor/ml/code/utils.py
Path: /home/kevin/Dropbox (LSMO)/proj75_mofcolor/ml/code
Created Date: Monday, February 24th 2020, 4:42:56 pm
Author: Kevin Jablonka

Copyright (c) 2020 Kevin Jablonka
"""

import os
import collections
import pickle
import random
import numpy as np
import matplotlib.patches as mpatch
from webcolors import rgb_to_hex
import matplotlib.pyplot as plt
from comet_ml import Experiment
import time
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras import backend as BK
import pandas as pd
import ruamel.yaml as yaml
from .descriptornames import *
from numpy.random import seed
import joblib


def augment_data(
    df, augment_dict, r_col="r", g_col="g", b_col="b", name_col="color_cleaned"
):
    df_ = df.copy()

    new_rows = []
    for i, row in df.iterrows():
        color = row[name_col]
        r_ = row.copy()
        for rgb in augment_dict[color]:
            r_[r_col] = rgb[0]
            r_[g_col] = rgb[1]
            r_[b_col] = rgb[2]
            new_rows.append(r_)

    new_df = pd.concat([df, pd.DataFrame(new_rows)])
    return new_df


def augment_random(colorname, augment_dict):
    possible_colors = augment_dict[colorname]
    chosen_color = random.choice(possible_colors)
    return np.array(chosen_color)


def tf_augment_random():
    raise NotImplementedError


def read_pickle(file):
    with open(file, "rb") as fh:
        result = pickle.load(fh)
    return result


def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = (
        tf.abs(error) < 1
    )  # replace `tf` with `K` where `K = keras.backend`
    squared_loss = tf.square(error) / 2  # replace `tf` with `K`
    linear_loss = tf.abs(error) - 0.5  # replace `tf` with `K`
    return tf.where(is_small_error, squared_loss, linear_loss)


def get_timestamp_string():
    t = time.localtime()
    timestamp = time.strftime("%b-%d-%Y_%H%M", t)
    return timestamp


def mapping_to_target_range(x, target_min=0, target_max=1):
    """Linear activation function that constrains output to a range
    
    Arguments:
        x {tensor} -- input tensor
    
    Keyword Arguments:
        target_min {float} -- minimum output (default: {0})
        target_max {float} -- maximum output (default: {1})
    
    Returns:
        tensor -- constrained linear activation
    """
    x02 = x + 1  # x in range(0,2)
    scale = (target_max - target_min) / 2.0
    return x02 * scale + target_min


def mapping_to_target_range_sig(x, target_min=0, target_max=255):
    """Sigmoid activation function that constrains output to a range
    
    Arguments:
        x {tensor} -- input tensor
    
    Keyword Arguments:
        target_min {float} -- minimum output (default: {0})
        target_max {float} -- maximum output (default: {255})
    
    Returns:
        tensor -- constrained Sigmoid activation
    """
    x02 = BK.sigmoid(x) + 1  # x in range(0,2)
    scale = (target_max - target_min) / 2.0
    return x02 * scale + target_min


def plot_predictions(predictions, labels, names, sample=100, outname=None):
    """Plot figure that compares color of predictions versus acutal colors.
    
    Arguments:
        predictions {iterable} -- iterable of rgb colors
        labels {iterable} -- iterable of rgb colors
        names {iterable} -- iterable of strings
    
    Keyword Arguments:
        sample {int} -- how many samples to plot (default: {100})
        outname {string} -- path to which figure is saved (default: {None})
    """
    fig = plt.figure(figsize=[4.8, 16])
    ax = fig.add_axes([0, 0, 1, 1])

    predictions = predictions[:sample]
    names = names[:sample]
    labels = labels[:sample]

    predictions = [rgb_to_hex((int(c[0]), int(c[1]), int(c[2]))) for c in predictions]
    true = [rgb_to_hex((int(c[0]), int(c[1]), int(c[2]))) for c in labels]

    for i in range(len(predictions)):
        r1 = mpatch.Rectangle((0, i), 1, 1, color=predictions[i])
        r2 = mpatch.Rectangle((1, i), 1, 1, color=true[i])
        txt = ax.text(2, i + 0.5, "  " + names[i], va="center", fontsize=10)

        ax.add_patch(r1)
        ax.add_patch(r2)
        ax.axhline(i, color="k")

    ax.text(0.5, i + 1.5, "prediction", ha="center", va="center")
    ax.text(1.5, i + 1.5, "median RGB for label", ha="center", va="center")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, i + 2)
    ax.axis("off")

    fig.tight_layout()

    if outname is not None:
        fig.savefig(outname, bbox_inches="tight")


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def parse_config(yml_file):
    with open(yml_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def make_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def select_features(selection: list):
    selected_features = []
    for feature in selection:
        selected_features.extend(descriptor_dict[feature])

    return selected_features
