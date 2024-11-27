import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import os

from pm4py import write_xes

from nirdizati_light.log.common import get_log, split_train_val_test
from nirdizati_light.encoding.common import get_encoded_df, EncodingType, decode_df
from nirdizati_light.encoding.constants import TaskGenerationType, PrefixLengthStrategy, EncodingTypeAttribute
from nirdizati_light.encoding.time_encoding import TimeEncodingType
from nirdizati_light.labeling.common import LabelTypes

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline

from nirdizati_light.oversampling.common import get_sampling_strategy, AugmentationStrategy, generate_traces_smotenc

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

CONF = {
    # path to input and output folder
    'input':  os.path.join('..', 'input_data'),
    'output': os.path.join('..', 'output_data'),
    'subfolder': os.path.join('data_augmentation_smotenc', 'xes'),
    'log_name': 'Consulta_full.xes',
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

    # augmentation strategy
    'augmentation_strategy': AugmentationStrategy.FULL_PERC.value,
    'augmentation_factor_list': [0.05, 0.1, 0.15],
    
    'target_event': None,
    'seed': SEED,
}

print('Loading log...')

# define input and output path
input_log = os.path.join(CONF['input'], CONF['subfolder'], CONF['log_name'])
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



for augmentation_factor in CONF['augmentation_factor_list']:
    print('file: %s, augmentation: %s/%s' % (
    CONF['log_name'], CONF['augmentation_factor_list'].index(augmentation_factor) + 1, len(CONF['augmentation_factor_list'])))
    # define output path
    suffix = str(augmentation_factor).replace('0.', '')
    output_log = os.path.join(CONF['output'], CONF['subfolder'], CONF['log_name'].replace('.xes', '_smotenc_' + suffix + '.xes'))

    # use SMOTENC to augment train_df
    new_df = generate_traces_smotenc(
        df=train_df, encoder=encoder,
        augmentation_strategy=CONF['augmentation_strategy'],
        augmentation_factor=0.1,
        random_state=CONF['seed'],
    )

    # decode dataframe to log
    new_log = decode_df(
        df=new_df,
        encoder=encoder,
        log=log,
        feature_encoding_type=CONF['feature_selection'],
    )

    # export log
    write_xes(new_log, output_log)

print('Fin')

