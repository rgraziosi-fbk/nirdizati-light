from enum import Enum

import hyperopt
import numpy as np
from hyperopt import Trials, hp, fmin
from hyperopt.pyll import scope

from nirdizati_light.predictive_model.common import ClassificationMethods


class HyperoptTarget(Enum):
    AUC = 'auc'
    F1 = 'f1_score'
    MAE = 'mae'


def _get_space(model_type) -> dict:
    if model_type is ClassificationMethods.RANDOM_FOREST.value:
        return {
            'n_estimators': hp.choice('n_estimators', np.arange(150, 1000, dtype=int)),
            'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', 'auto', None]),
            'criterion': hp.choice('criterion',['gini','entropy']),
            'warm_start': True
        }

    elif model_type is ClassificationMethods.KNN.value:
        return {
            'n_neighbors': hp.choice('n_neighbors', np.arange(1, 20, dtype=int)),
            'weights': hp.choice('weights', ['uniform', 'distance']),
        }

    elif model_type is ClassificationMethods.XGBOOST.value:
        return {
            'n_estimators': hp.choice('n_estimators', np.arange(150, 1000, dtype=int)),
            'max_depth': scope.int(hp.quniform('max_depth', 3, 30, 1)),
        }

    elif model_type is ClassificationMethods.SGDCLASSIFIER.value:
        return {
            'loss': hp.choice('loss', ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error',
                                       'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
            'penalty': hp.choice('penalty', [None, 'l1', 'l2', 'elasticnet']),
            'alpha': hp.uniform('alpha', 0.0001, 0.5),
            'fit_intercept': hp.choice('fit_intercept', [True, False]),
            'tol': hp.uniform('tol', 1e-3, 0.5),
            'shuffle': hp.choice('shuffle', [True, False]),
            'eta0': hp.quniform('eta0', 0, 2),
            # 'early_stopping': hp.choice('early_stopping', [True, False]), #needs to be false with partial_fit
            'validation_fraction': 0.1,
            'n_iter_no_change': scope.int(hp.quniform('n_iter_no_change', 1, 30, 5))
        }
    elif model_type is ClassificationMethods.SVM.value:
        return{
            'kernel' : hp.choice('kernel',['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']),
            'C' : hp.uniform('C',0.1,1)
        }
    elif model_type is ClassificationMethods.PERCEPTRON.value:
        return {
            'penalty': hp.choice('penalty', [None, 'l1', 'l2', 'elasticnet']),
            'alpha': hp.uniform('alpha', 0.0001, 0.5),
            'fit_intercept': hp.choice('fit_intercept', [True, False]),
            'tol': hp.uniform('tol', 1e-3, 0.5),
            'shuffle': hp.choice('shuffle', [True, False]),
            'eta0': scope.int(hp.quniform('eta0', 4, 30, 1)),
            # 'early_stopping': hp.choice('early_stopping', [True, False]), #needs to be false with partial_fit
            'validation_fraction': 0.1,
            'n_iter_no_change': scope.int(hp.quniform('n_iter_no_change', 5, 30, 5))
        }
    elif model_type is ClassificationMethods.MLP.value:
        return {
            'hidden_layer_sizes': scope.int(hp.uniform('hidden_layer_sizes',10,100)),
            'alpha': hp.uniform('alpha', 0.0001, 0.5),
            'shuffle': hp.choice('shuffle', [True, False]),
#            'eta0': scope.int(hp.quniform('eta0', 4, 30, 1)),
            # 'early_stopping': hp.choice('early_stopping', [True, False]), #needs to be false with partial_fit
            'validation_fraction': 0.1,
            'n_iter_no_change': scope.int(hp.quniform('n_iter_no_change', 5, 30, 5))
        }
    elif model_type is ClassificationMethods.RANDOM_FOREST.value:
        return {
            'n_estimators': hp.choice('n_estimators', np.arange(150, 1000, dtype=int)),
            'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', 'auto', None]),
            'warm_start': True
        }

    elif model_type is ClassificationMethods.LSTM.value:
        return {
            'activation': hp.choice('activation', ['linear', 'tanh', 'relu']),
            'kernel_initializer': hp.choice('kernel_initializer', ['glorot_uniform']),
            'optimizer': hp.choice('optimizer', ['adam', 'nadam', 'rmsprop'])
        }

    else:
        raise Exception('Unsupported model_type')


def retrieve_best_model(predictive_model, model_type, max_evaluations, target, seed=None):
    """
    Perform hyperparameter optimization on the given model

    Parameters:
    predictive_model: the model to perform optimization on (must be of class PredictiveModel)
    model_type: type of predictive model (can be one of predictive_model.common.ClassificationMethods)
    max_evaluations: maximum number of hyperparameter configurations to try
    target: which target score to optimize for (can be on of hyperparameter_optimisation.common.HyperoptTarget)
    seed: optional seed value for reproducibility

    Return: a tuple containing both the best model and the best hyperparameter configuration
    """

    space = _get_space(model_type)
    trials = Trials()

    fmin(
        lambda x: predictive_model.train_and_evaluate_configuration(config=x, target=target),
        space,
        algo=hyperopt.tpe.suggest,
        max_evals=max_evaluations,
        trials=trials,rstate=seed
    )
    best_candidate = trials.best_trial['result']

    return best_candidate['model'], best_candidate['config']
