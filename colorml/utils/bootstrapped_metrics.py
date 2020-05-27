# -*- coding: utf-8 -*-
"""Small CLI utility to get some bootstrapped metrics, this assumes that all preprocessing on the data already has been carried out or will be carried out upon the predict call to the model"""

from __future__ import absolute_import

from typing import Union

import click
import joblib
import numpy as np
import pandas as pd
import tqdm
from comet_ml import Experiment
from six.moves import range
from sklearn.metrics import (mean_absolute_error, mean_squared_error, median_absolute_error, r2_score)


def get_metrics(model, X: np.array, y_true: np.array) -> dict:
    prediction = model.predict(X)
    r2 = r2_score(y_true, prediction)
    mae = mean_absolute_error(y_true, prediction)
    mse = mean_squared_error(y_true, prediction)
    mdae = median_absolute_error(y_true, prediction)

    return {'r2': r2, 'mae': mae, 'mse': mse, 'mdae': mdae}


def get_bootstrap_sample(X: np.array, y: np.array) -> Union[np.array, np.array]:
    indices = np.arange(0, len(X) - 1)
    sample = np.random.choice(indices, len(indices), replace=True)

    return X[sample], y[sample]


def get_metrics(model, X: np.array, y_true: np.array, num_rounds: int = 1000) -> list:
    metrics = []

    # pretty inefficient way of doing it ... should probably at least use concurrent.futures
    for _ in tqdm(list(range(num_rounds))):
        X_sample, y_sample = get_bootstrap_sample(X, y_true)
        metrics.append(get_metrics(model, X_sample, y_sample))

    return metrics


@click.command('cli')
@click.argument('modelpath', type=click.Path(exists=True))
@click.argument('Xpath', type=click.Path(exists=True))
@click.argument('ypath', type=click.Path(exists=True))
@click.argument('outname', type=click.Path())
def main(modelpath, Xpath, ypath, outname):
    model = joblib.load(modelpath)
    X = np.load(Xpath)
    y = np.load(ypath)

    metrics = get_metrics(model, X, y)

    df = pd.DataFrame(metrics)
    df.to_csv(outname, index=False)


if __name__ == '__main__':
    main()
