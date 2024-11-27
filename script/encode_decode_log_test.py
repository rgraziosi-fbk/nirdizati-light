import os
import random
import numpy as np
import pandas as pd
import os

from nirdizati_light.log.common import get_log, split_train_val_test
from nirdizati_light.encoding.common import get_encoded_df, EncodingType, decode_df
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

CONF = {
    # path to input and output folder
    'input':  os.path.join('..', 'input_data'),
    'output': os.path.join('..', 'output_data'),
    'subfolder': os.path.join('data_augmentation_smotenc', 'xes'),
    'log_name': 'SynLoan_full.xes',
    # train-validation-test set split percentages
    'train_val_test_split': [0.7, 0.15, 0.15],
    # prefix length truncation
    'prefix_length_strategy': PrefixLengthStrategy.PERCENTAGE.value,
    'prefix_length': 1,  # 1 mean full traces

    # whether to use padding or not in encoding
    'padding': True,
    # which encoding to use
    'feature_selection': EncodingType.COMPLEX.value,
    # which attribute encoding to use
    'attribute_encoding': EncodingTypeAttribute.LABEL.value,
    # which time encoding to use
    'time_encoding': TimeEncodingType.NONE.value,

    # the label to be predicted (e.g. outcome, next activity)
    'labeling_type': LabelTypes.ATTRIBUTE_STRING.value,
    # whether the model should be trained on the specified prefix length (ONLY_THIS) or to every prefix in range [1, prefix_length] (ALL_IN_ONE)
    'task_generation_type': TaskGenerationType.ONLY_THIS.value,
    
    'target_event': None,
    'seed': SEED,
}

print('Loading log...')

# define input and output path
input_log = os.path.join(CONF['input'], CONF['subfolder'], CONF['log_name'])
output_log = os.path.join(CONF['input'], CONF['subfolder'], CONF['log_name'].replace('.xes', '_smotenc.xes'))

# load the log
log = get_log(filepath=input_log)

# create encoder
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

# split encoded df
print('Splitting in train, validation and test...')
train_size, val_size, test_size = CONF['train_val_test_split']
train_df, val_df, test_df = split_train_val_test(full_df, train_size, val_size, test_size, shuffle=False, seed=CONF['seed'])

# try to decode
new_log = decode_df(df=train_df, encoder=encoder, log=log, feature_encoding_type=CONF['feature_selection'],)

print('here')

