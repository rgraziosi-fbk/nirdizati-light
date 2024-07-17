import os
import random
import numpy as np
import pandas as pd
from hyperopt import hp
from hyperopt.pyll import scope

from nirdizati_light.log.common import get_log, split_train_val_test
from nirdizati_light.encoding.common import get_encoded_df, EncodingType
from nirdizati_light.encoding.constants import TaskGenerationType, PrefixLengthStrategy, EncodingTypeAttribute
from nirdizati_light.encoding.time_encoding import TimeEncodingType
from nirdizati_light.labeling.common import LabelTypes
from nirdizati_light.predictive_model.common import ClassificationMethods
from nirdizati_light.predictive_model.predictive_model import PredictiveModel
from nirdizati_light.hyperparameter_optimisation.common import retrieve_best_model, HyperoptTarget
from nirdizati_light.evaluation.common import evaluate_classifier,evaluate_classifiers, plot_model_comparison_classification
from nirdizati_light.explanation.common import ExplainerType, explain

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

CONF = {
    # path to log
    'data': os.path.join('datasets', 'BPIC11_f1.csv'),
    # train-validation-test set split percentages
    'train_val_test_split': [0.7, 0.1, 0.2],

    # path to output folder
    'output': 'output_data',

    'prefix_length_strategy': PrefixLengthStrategy.FIXED.value,
    'prefix_length': 15,

    # whether to use padding or not in encoding
    'padding': True,
    # which encoding to use
    'feature_selection': EncodingType.SIMPLE_TRACE.value,
    # which attribute encoding to use
    'attribute_encoding': EncodingTypeAttribute.LABEL.value,
    # which time encoding to use
    'time_encoding': TimeEncodingType.DATE_AND_DURATION.value,

    # the label to be predicted (e.g. outcome, next activity)
    'labeling_type': LabelTypes.ATTRIBUTE_STRING.value,
    # whether the model should be trained on the specified prefix length (ONLY_THIS) or to every prefix in range [1, prefix_length] (ALL_IN_ONE)
    'task_generation_type': TaskGenerationType.ONLY_THIS.value,
    
    # list of predictive models and their respective hyperparameter optimization space
    # if it is None, then the default hyperopt space will be used; otherwise, the provided space will be used
    'predictive_models': {
        ClassificationMethods.KNN.value: None,
        ClassificationMethods.RANDOM_FOREST.value: {
            'n_estimators': hp.choice('n_estimators', np.arange(200, 500, dtype=int)),
            'max_depth': scope.int(hp.quniform('max_depth', 5, 30, 1)),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
            'criterion': hp.choice('criterion', ['gini','entropy']),
        },
        ClassificationMethods.LSTM.value: {
            'max_num_epochs': 200,
            'lr': 3e-4,
            'lstm_hidden_size': hp.choice('lstm_hidden_size', np.arange(100, 500, dtype=int)),
            'lstm_num_layers': hp.choice('lstm_num_layers', np.arange(1, 3, dtype=int)),
            'early_stop_patience': scope.int(hp.quniform('early_stop_patience', 20, 50, 5)),
            'early_stop_min_delta': hp.uniform('early_stop_min_delta', 0.005, 0.05),
            'batch_size': 128
        },
        ClassificationMethods.CUSTOM_PYTORCH.value: {
            'max_num_epochs': 200,
            'lr': 3e-4,
            'lstm_hidden_size': hp.choice('lstm_hidden_size', np.arange(100, 500, dtype=int)),
            'lstm_num_layers': hp.choice('lstm_num_layers', np.arange(1, 3, dtype=int)),
            'early_stop_patience': scope.int(hp.quniform('early_stop_patience', 20, 50, 5)),
            'early_stop_min_delta': hp.uniform('early_stop_min_delta', 0.005, 0.05),
            'batch_size': 128
        }
    },
    
    # which metric to optimize hyperparameters for
    'hyperparameter_optimisation_target': HyperoptTarget.F1.value,
    # number of hyperparameter configurations to try
    'hyperparameter_optimisation_evaluations': 3,

    # explainability method to use
    'explanator': ExplainerType.DICE.value,
    
    'target_event': None,
    'seed': SEED,
}

print('Loading log...')
log = get_log(filepath=CONF['data'], separator=';')

print('Encoding traces...')
encoder, full_df = get_encoded_df(
  log=log,
  feature_encoding_type=CONF['feature_selection'],
  prefix_length=CONF['prefix_length'],
  prefix_length_strategy=CONF['prefix_length_strategy'],
  time_encoding_type=CONF['time_encoding'],
  attribute_encoding=CONF['attribute_encoding'],
  padding=CONF['padding'],
  labeling_type=CONF['labeling_type'],
  task_generation_type=CONF['task_generation_type'],
  target_event=CONF['target_event'],
)

print('Splitting in train, validation and test...')
train_size, val_size, test_size = CONF['train_val_test_split']
train_df, val_df, test_df = split_train_val_test(full_df, train_size, val_size, test_size, shuffle=False, seed=CONF['seed'])

print('Instantiating predictive models...')
predictive_models = []

for predictive_model_type, predictive_model_hyperopt_space in CONF['predictive_models'].items():
    custom_model_class = None

    # specify which custom Pytorch model to use
    if predictive_model_type is ClassificationMethods.CUSTOM_PYTORCH.value:
        from custom_model_example import CustomModelExample
        custom_model_class = CustomModelExample

    predictive_models.append(
        PredictiveModel(
            predictive_model_type,
            train_df,
            val_df,
            test_df,
            prefix_length=CONF['prefix_length'],
            hyperopt_space=predictive_model_hyperopt_space,
            custom_model_class=custom_model_class
        )
    )

print('Running hyperparameter optimization...')
best_candidates,best_model_idx, best_model_model, best_model_config = retrieve_best_model(
    predictive_models,
    max_evaluations=CONF['hyperparameter_optimisation_evaluations'],
    target=CONF['hyperparameter_optimisation_target']
)

best_model = predictive_models[best_model_idx]
best_model.model = best_model_model
best_model.config = best_model_config
print(f'Best model is {best_model.model_type}')


print('Evaluating best model...')
predicted, scores = best_model.predict(test=True)
actual = test_df['label']

initial_result = evaluate_classifier(actual, predicted, scores)
results = evaluate_classifiers(predictive_models,actual)
plot_model_comparison_classification(results)
print(f'Evaluation: {initial_result}')

print('Computing explanation...')
test_df_correct = test_df[(test_df['label'] == predicted) & (test_df['label'] == 0)]
cf_dataset = pd.concat([train_df, val_df], ignore_index=True)
full_df = pd.concat([train_df, val_df, test_df])
cf_dataset.loc[len(cf_dataset)] = 0

dice_results = explain(CONF, best_model, encoder=encoder, df=full_df.iloc[:, 1:],
        query_instances=test_df_correct.iloc[:, 1:],
        method='multi_objective_genetic', optimization='multiobjective',
        heuristic='heuristic_2', support=0.95,
        timestamp_col_name='Complete Timestamp', # name of the timestamp column in the log
        model_path='./experiments/process_models/process_models',
        random_seed=CONF['seed'], adapted=True, filtering=False)

print('Explanation:',dice_results)