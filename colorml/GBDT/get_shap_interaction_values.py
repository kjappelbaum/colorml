# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import click
import joblib
import numpy as np
import shap

np.random.seed(821996)

model_median = joblib.load('regressor_medianrun_2020_09_10_13_19_1599736778False.joblib')

X_test = np.load('X_test_run_2020_09_10_13_19_1599736778.npy')
X_train = np.load('X_train_run_2020_09_10_13_19_1599736778.npy')
y_test = np.load('y_test_run_2020_09_10_13_19_1599736778.npy')
y_train = np.load('y_train_run_2020_09_10_13_19_1599736778.npy')

samples = X_train[np.random.choice(X_train.shape[0], 2000, replace=False)]


@click.command('cli')
@click.argument('i')
def main(i):
    print('starting the computation')
    np.save('samples_{}'.format(i), samples)

    interaction_values = []

    for i, estimator in enumerate(model_median.estimators_):
        if i < 6:
            booster = estimator.estimators_[int(i)]
            explainer = shap.TreeExplainer(booster)
            print('getting interaction values')
            shap_interaction_values_ = explainer.shap_interaction_values(samples)
            print('Done, saving')
            interaction_values.append(shap_interaction_values_)

    np.save('shap_interaction_{}'.format(i), interaction_values)
    joblib.dump(explainer, 'explainer_{}.joblib'.format(i))


if __name__ == '__main__':
    main()
