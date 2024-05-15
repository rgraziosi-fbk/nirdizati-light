from enum import Enum
import torch
import hyperopt
import numpy as np
from hyperopt import Trials, hp, fmin
from hyperopt.pyll import scope

from nirdizati_light.predictive_model.common import ClassificationMethods, RegressionMethods


class HyperoptTarget(Enum):
    AUC = 'auc'
    F1 = 'f1_score'
    MAE = 'mae'
    ACCURACY = 'accuracy'
    MCC = 'mcc'
    MAPE = 'mape'
    RMSE= 'rmse'


def _get_space(model_type) -> dict:
    if model_type is ClassificationMethods.RANDOM_FOREST.value:
        return {
            'n_estimators': hp.choice('n_estimators', np.arange(150, 1000, dtype=int)),
            'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
            'criterion': hp.choice('criterion',['gini','entropy']),
            'warm_start': True
        }
    elif model_type is ClassificationMethods.DT.value:
        return {
            'max_depth': hp.choice('max_depth', range(1, 6)),
            'max_features': hp.choice('max_features', range(7, 50)),
            'criterion': hp.choice('criterion', ["gini", "entropy"]),
            'min_samples_split': hp.choice('min_samples_split', range(2, 10)),
            'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 10)),

        }
    elif model_type is ClassificationMethods.KNN.value:
        return {
            'n_neighbors': hp.choice('n_neighbors', np.arange(1, 20, dtype=int)),
            'weights': hp.choice('weights', ['uniform', 'distance']),
        }

    elif model_type is ClassificationMethods.XGBOOST.value:
        return {
            'n_estimators': hp.choice('n_estimators', np.arange(150, 1000, dtype=int)),
        'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
            'max_depth': scope.int(hp.quniform('max_depth', 2, 30, 1)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
            'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
            'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        }
    elif model_type is RegressionMethods.XGBOOST.value:
        return  {
                #'n_estimators': hp.choice('n_estimators', range(10, 500)),
                #'max_depth': scope.int(hp.quniform('max_depth', 2, 30, 1)),
                #'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
                #'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
                #'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                #'gamma': hp.uniform('gamma', 0, 5),
                'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart', 'goss']),
                'objective': hp.choice('objective', ['reg:squarederror', 'reg:squaredlogerror','reg:linear']),
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
            'eta0': hp.quniform('eta0', 0, 5, 1),
            # 'early_stopping': hp.choice('early_stopping', [True, False]), #needs to be false with partial_fit
            'validation_fraction': 0.1,
            'n_iter_no_change': scope.int(hp.quniform('n_iter_no_change', 1, 30, 5))
        }
    elif model_type is RegressionMethods.SGDREGRESSOR.value:
        return {
            'loss': hp.choice('loss', ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error',
                                       'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
            'penalty': hp.choice('penalty', [None, 'l1', 'l2', 'elasticnet']),
            'alpha': hp.uniform('alpha', 0.0001, 0.5),
            'fit_intercept': hp.choice('fit_intercept', [True, False]),
            'tol': hp.uniform('tol', 1e-3, 0.5),
            'shuffle': hp.choice('shuffle', [True, False]),
            'eta0': hp.quniform('eta0', 0, 5, 1),
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
    elif model_type is RegressionMethods.RANDOM_FOREST.value:
        return {
            'n_estimators': hp.choice('n_estimators', np.arange(150, 1000, dtype=int)),
            'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', 'auto', None]),
            'warm_start': True
        }

    elif model_type is ClassificationMethods.LSTM.value:
        return {
            'lr': hp.uniform('lr', 0.0001, 0.1),
            'lstm_hidden_size': hp.choice('lstm_hidden_size', np.arange(50, 200, dtype=int)),
            'lstm_num_layers': hp.choice('lstm_num_layers', np.arange(1, 3, dtype=int)),
        }

    else:
        raise Exception('Unsupported model_type')


def retrieve_best_model(predictive_models, max_evaluations, target, seed=None):
    """
    Perform hyperparameter optimization on the given model

    :param list predictive_models: list of models to perform optimization on (each model must be of class PredictiveModel)
    :param int max_evaluations: maximum number of hyperparameter configurations to try
    :param nirdizati_light.hyperparameter_optimisation.common.HyperoptTarget target: which target score to optimize for
    :param int seed: optional seed value for reproducibility

    :return: a tuple containing the best candidates, the best model index, the best model and the best hyperparameter configuration
    """

    best_candidates = []
    best_target_per_model = []

    for predictive_model in predictive_models:
        print(f'Running hyperparameter optimization on model {predictive_model.model_type}...')

        space = _get_space(predictive_model.model_type)
        trials = Trials()

        fmin(
            lambda x: predictive_model.train_and_evaluate_configuration(config=x, target=target),
            space,
            algo=hyperopt.tpe.suggest,
            max_evals=max_evaluations,
            trials=trials,rstate=np.random.default_rng(seed)
        )
        best_candidate = trials.best_trial['result']

        best_candidates.append(best_candidate)
        best_target_per_model.append(best_candidate['result'][target])

    # Find the best performing model
    best_model_idx = best_target_per_model.index(max(best_target_per_model))

    return best_candidates, best_model_idx, best_candidates[best_model_idx]['model'], best_candidates[best_model_idx]['config']
