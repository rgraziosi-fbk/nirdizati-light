from enum import Enum
from typing import Optional, Union
from predictive_model.predictive_model import PredictiveModel
import hyperopt
import numpy as np
from hyperopt import Trials, hp, fmin
from hyperopt.pyll import scope

from nirdizati_light.predictive_model.common import ClassificationMethods, RegressionMethods


class HyperoptTarget(Enum):
    """
    Available targets for hyperparameter optimisation
    """
    AUC = 'auc'
    F1 = 'f1_score'
    MAE = 'mae'
    ACCURACY = 'accuracy'


def _get_space(model_type: Union[ClassificationMethods, RegressionMethods]) -> dict:
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
    elif model_type is ClassificationMethods.RANDOM_FOREST.value:
        return {
            'n_estimators': hp.choice('n_estimators', np.arange(150, 1000, dtype=int)),
            'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', 'auto', None]),
            'warm_start': True
        }

    elif model_type in [ClassificationMethods.LSTM.value, ClassificationMethods.CUSTOM_PYTORCH.value]:
        return {
            'max_num_epochs': 200,
            'lr': hp.uniform('lr', 0.0001, 0.1),
            'lstm_hidden_size': hp.choice('lstm_hidden_size', np.arange(50, 200, dtype=int)),
            'lstm_num_layers': hp.choice('lstm_num_layers', np.arange(1, 3, dtype=int)),
            'early_stop_patience': scope.int(hp.quniform('early_stop_patience', 5, 30, 5)),
            'early_stop_min_delta': hp.uniform('early_stop_min_delta', 0.005, 0.05),
            'batch_size': 128
        }
    elif model_type is RegressionMethods.RANDOM_FOREST.value:
        return {
            'n_estimators': hp.choice('n_estimators', np.arange(150, 1000, dtype=int)),
            'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
            'criterion': hp.choice('criterion', ['squared_error', 'friedman_mse', 'poisson', 'absolute_error']),
            'warm_start': True
        }
    else:
        raise Exception('Unsupported model_type')


def retrieve_best_model(
    predictive_models: list[PredictiveModel],
    max_evaluations: int,
    target: HyperoptTarget,
    seed: Optional[int]=None
):
    """
    Perform hyperparameter optimization on the given model.

    Args:
        predictive_models (list[PredictiveModel]): List of models to perform optimization on (each model must be of class PredictiveModel).
        max_evaluations (int): Maximum number of hyperparameter configurations to try.
        target (HyperoptTarget): Which target score to optimize for.
        seed (Optional[int]): Optional seed value for reproducibility.

    Returns:
        tuple: A tuple containing the best candidates, the best model index, the best model, and the best hyperparameter configuration.
    """

    best_candidates = []
    best_target_per_model = []

    if type(predictive_models) is not list:
        predictive_models = [predictive_models]

    for predictive_model in predictive_models:
        print(f'Running hyperparameter optimization on model {predictive_model.model_type}...')

        space = _get_space(predictive_model.model_type)

        # update hyperopt space with the user provided one, if available
        if predictive_model.hyperopt_space is not None:
            space.update(predictive_model.hyperopt_space)

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
