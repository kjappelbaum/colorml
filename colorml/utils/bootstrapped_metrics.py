# -*- coding: utf-8 -*-
"""Small CLI utility to get some bootstrapped metrics, this assumes that all preprocessing on the data already has been carried out or will be carried out upon the predict call to the model"""

from __future__ import absolute_import

import os
from typing import Union

import click
import joblib
import numpy as np
import pandas as pd
from comet_ml import Experiment
from six.moves import range
from sklearn.metrics import (mean_absolute_error, mean_squared_error, median_absolute_error, r2_score)
from tqdm import tqdm


def get_metrics_dict(model, X: np.array, y_true: np.array, experiment) -> dict:
    with experiment.test():
        prediction = model.predict(X)

        r2 = r2_score(y_true, prediction)
        mae = mean_absolute_error(y_true, prediction)
        mse = mean_squared_error(y_true, prediction)
        mdae = median_absolute_error(y_true, prediction)

    return {'r2': r2, 'mae': mae, 'mse': mse, 'mdae': mdae}


def get_bootstrap_sample(X: np.array, y: np.array) -> Union[np.array, np.array]:
    indices = np.arange(0, len(X) - 1)
    sample = np.random.choice(indices, len(y), replace=True)

    return X[sample, :], y[sample, :]


def get_metrics(model, X: np.array, y_true: np.array, experiment, num_rounds: int = 5000) -> list:
    metrics = []

    # pretty inefficient way of doing it ... should probably at least use concurrent.futures
    for _ in tqdm(range(num_rounds)):
        X_sample, y_sample = get_bootstrap_sample(X, y_true)
        metrics.append(get_metrics_dict(model, X_sample, y_sample, experiment))

    return metrics


@click.command('cli')
@click.argument('modelpath', type=click.Path(exists=True))
@click.argument('xpath', type=click.Path(exists=True))
@click.argument('ypath', type=click.Path(exists=True))
@click.argument('outname', type=click.Path())
def main(modelpath, xpath, ypath, outname):
    experiment = Experiment(api_key=os.environ['COMET_API_KEY'], project_name='color-ml')
    model = joblib.load(modelpath)
    X = np.load(xpath)
    y = np.load(ypath)

    metrics = get_metrics(model, X, y, experiment)

    df = pd.DataFrame(metrics)
    df.to_csv(outname, index=False)
    experiment.log_asset(outname)


if __name__ == '__main__':
    main()
