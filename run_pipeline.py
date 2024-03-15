import os
import random
import numpy as np
import pandas as pd

from nirdizati_light.log.common import get_log
from nirdizati_light.encoding.common import get_encoded_df, EncodingType
from nirdizati_light.encoding.constants import TaskGenerationType, PrefixLengthStrategy, EncodingTypeAttribute
from nirdizati_light.encoding.time_encoding import TimeEncodingType
from nirdizati_light.labeling.common import LabelTypes
from nirdizati_light.predictive_model.common import ClassificationMethods
from nirdizati_light.predictive_model.predictive_model import PredictiveModel
from nirdizati_light.hyperparameter_optimisation.common import retrieve_best_model, HyperoptTarget
from nirdizati_light.evaluation.common import evaluate_classifier,evaluate_classifiers,plot_model_comparison
from nirdizati_light.explanation.common import ExplainerType, explain

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

LOG_NAME = 'bpic2012_O_ACCEPTED-COMPLETE'

CONF = {
    'data': os.path.join('..','datasets', LOG_NAME, 'full.xes'),         # path to log
    'train_val_test_split': [0.7, 0.15, 0.15],                      # train-validation-test set split percentages

    'output': 'output_data',                                        # path to output folder

    'prefix_length_strategy': PrefixLengthStrategy.FIXED.value,     #
    'prefix_length': 20,                                            # 

    'padding': True,                                                # whether to use padding or not in encoding
    'feature_selection': EncodingType.SIMPLE_TRACE.value,           # which encoding to use
    'attribute_encoding': EncodingTypeAttribute.LABEL.value,        # which attribute encoding to use
    'time_encoding': TimeEncodingType.NONE.value,                   # which time encoding to use

    'task_generation_type': TaskGenerationType.ONLY_THIS.value,     #
    'labeling_type': LabelTypes.ATTRIBUTE_STRING.value,             # 
    
    'predictive_models': [                                          # list of predictive models to train
         ClassificationMethods.RANDOM_FOREST.value,
        #ClassificationMethods.KNN.value,
        # ClassificationMethods.LSTM.value,
         ClassificationMethods.MLP.value,
         ClassificationMethods.PERCEPTRON.value,
         ClassificationMethods.SGDCLASSIFIER.value,
        # ClassificationMethods.SVM.value,
         ClassificationMethods.XGBOOST.value,
    ],
    
    'hyperparameter_optimisation_target': HyperoptTarget.F1.value,  # which metric to optimize hyperparameters for
    'hyperparameter_optimisation_evaluations': 15,                  # number of hyperparameter configurations to try

    'explanator': ExplainerType.DICE.value,                         # explainability method to use
    
    'target_event': None,
    'seed': SEED,
}

print('Loading log...')
log = get_log(filepath=CONF['data'])

print('Encoding traces...')
encoder, full_df = get_encoded_df(log=log, CONF=CONF)

print('Splitting in train, validation and test...')
train_size, val_size, test_size = CONF['train_val_test_split']
train_df, val_df, test_df = np.split(full_df,[int(train_size*len(full_df)), int((train_size+val_size)*len(full_df))])

print('Instantiating predictive models...')
predictive_models = [PredictiveModel(CONF, predictive_model, train_df, val_df) for predictive_model in CONF['predictive_models']]

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
predicted, scores = best_model.predict()
actual = test_df['label']

initial_result = evaluate_classifier(actual, predicted, scores)
results = evaluate_classifiers(best_candidates, test_df,actual)
plot_model_comparison(results)
print(f'Evaluation: {initial_result}')

print('Computing explanation...')
test_df_correct = test_df[(test_df['label'] == predicted) & (test_df['label'] == 0)]
cf_dataset = pd.concat([train_df, val_df], ignore_index=True)
full_df = pd.concat([train_df, val_df, test_df])
cf_dataset.loc[len(cf_dataset)] = 0

explain(CONF, best_model, encoder=encoder, df=full_df.iloc[:, 1:],
        query_instances=test_df_correct.iloc[:, 1:],
        method='multi_objective_genetic', optimization='multiobjective',
        heuristic='heuristic_2', support=0.95,
        timestamp_col_name='Complete Timestamp', # name of the timestamp column in the log
        model_path='./experiments/process_models/process_models',
        random_seed=CONF['seed'], adapted=True, filtering=False)