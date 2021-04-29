from enum import Enum

import hyperopt
import numpy as np
from hyperopt import Trials, hp, fmin
from hyperopt.pyll import scope

from src.predictive_model.common import PredictionMethods


class HyperoptTarget(Enum):
    AUC = 'auc'
    F1 = 'f1_score'


def _get_space(model_type) -> dict:
    if model_type is PredictionMethods.RANDOM_FOREST.value:
        return {
            'n_estimators': hp.choice('n_estimators', np.arange(150, 1000, dtype=int)),
            'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', 'auto', None]),
            'warm_start': True
        }
    elif model_type is PredictionMethods.LSTM.value:
        return {
            'activation': hp.choice('activation', ['linear', 'tanh', 'relu']),
            'kernel_initializer': hp.choice('kernel_initializer', ['glorot_uniform']),
            'optimizer': hp.choice('optimizer', ['adam', 'nadam', 'rmsprop'])
        }
        # return {
        #     ### MANUALLY OPTIMISED PARAMS
        #     'n_estimators': 10,
        #     'max_depth': None,
        #     'max_features': 'auto',
        #     'n_jobs': -1,
        #     'random_state': 21,
        #     'warm_start': True
        #
        #     ### DEFAULT PARAMS
        #     # 'n_estimators': 100,
        #     # 'criterion': 'gini',
        #     # 'min_samples': 2,
        #     # 'min_samples_leaf': 1,
        #     # 'min_weight_fraction_leaf': 0.,
        #     # 'max_features': 'auto',
        #     # 'max_leaf_nodes': None,
        #     # 'min_impurity_decrease':0.,
        #     # 'min_impurity_split': 1e-7,
        #     # 'bootstrap': True,
        #     # 'oob_score': False,
        #     # 'n_jobs': None,
        #     # 'random_state': None,
        #     # 'verbose': 0,
        #     # 'warm_start': False,
        #     # 'class_weight': None,
        #     # 'ccp_alpha': 0.,
        #     # 'max_samples': None
        # }
    else:
        raise Exception('unsupported model_type')


def retrieve_best_model(predictive_model, model_type, max_evaluations, target):

    space = _get_space(model_type)
    trials = Trials()

    fmin(
        lambda x: predictive_model.train_and_evaluate_configuration(config=x, target=target),
        space,
        algo=hyperopt.tpe.suggest,
        max_evals=max_evaluations,
        trials=trials
    )
    best_candidate = trials.best_trial['result']

    return best_candidate['model'], best_candidate['config']
