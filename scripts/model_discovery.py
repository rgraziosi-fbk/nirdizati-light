import logging
import numpy as np
import sys
from DiCE import dice_ml as dice_ml
import sys
import os
import pickle
from src.predictive_model.common import ClassificationMethods, RegressionMethods, get_tensor, shape_label_df
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.spatial.distance import _validate_vector
from scipy.stats import median_abs_deviation
from datetime import datetime
import warnings
from src.encoding.common import get_encoded_df, EncodingType
from src.encoding.constants import TaskGenerationType, PrefixLengthStrategy, EncodingTypeAttribute
from src.encoding.time_encoding import TimeEncodingType
from src.evaluation.common import evaluate_classifier
from src.explanation.common import ExplainerType, explain
from src.explanation.wrappers.dice_wrapper import plausibility
from src.hyperparameter_optimisation.common import retrieve_best_model, HyperoptTarget
from src.labeling.common import LabelTypes
from src.log.common import get_log
import pm4py
from declare4py.src.declare4py.declare4py import Declare4Py
from declare4py.src.declare4py.enums import TraceState
from dataset_confs import DatasetConfs
from src.predictive_model.common import ClassificationMethods, get_tensor
from src.predictive_model.predictive_model import PredictiveModel, drop_columns
from src.explanation.visualizations.plot import line_plot,bar_plot
from DiCE import dice_ml
import pandas as pd
from dice_ml.utils import helpers
logger = logging.getLogger(__name__)
import warnings
import itertools
warnings.filterwarnings("ignore", category=UserWarning)

def model_discovery(encoder,df,dataset,features_names,prefix_length):
    d4py = Declare4Py()
    dataset_confs = DatasetConfs(dataset, where_is_the_file='')
    df = pd.DataFrame(df, columns=features_names)
    encoder.decode(df)
    df.insert(loc=0, column='Case ID', value=np.divmod(np.arange(len(df)), 1)[0] + 1)
    df.insert(loc=1, column='label', value=1)
    def transform(nested_list):
        regular_list = []
        for ele in nested_list:
            if type(ele) is list:
                regular_list.append(ele)
            else:
                regular_list.append([ele])
        return regular_list
    if encoder.feature_selection == 'complex':
        static_columns = list(dataset_confs.case_id_col.values()) + list(dataset_confs.label_col.values()) + list(
            dataset_confs.static_num_cols.values())
        dynamic_columns = list(dataset_confs.dynamic_cat_cols.values()) + list(
            dataset_confs.dynamic_num_cols.values()) + list(dataset_confs.timestamp_col.values())
        static_columns = transform(static_columns)
        dynamic_columns = transform(dynamic_columns)
        static_columns = list(itertools.chain(*static_columns))
        dynamic_columns = list(itertools.chain(*dynamic_columns))
        for i in range(len(dynamic_columns)):
            if dynamic_columns[i] == 'Activity':
                dynamic_columns[i] = 'prefix'
            elif dynamic_columns[i] == 'Activity Code':
                dynamic_columns[i] = 'prefix'
            elif dynamic_columns[i] == 'Activity code':
                dynamic_columns[i] = 'prefix'
        long_data = pd.wide_to_long(df, stubnames=dynamic_columns, i='Case ID',
                                    j='order', sep='_', suffix=r'\w+')
        long_data_sorted = long_data.sort_values(['Case ID', 'order'], ).reset_index(drop=False)

        for value in long_data_sorted[list(dataset_confs.timestamp_col.values())[0]]:
            if type(value) == float:
                long_data_sorted[list(dataset_confs.timestamp_col.values())[0]].replace(value,
                                                                                        datetime.fromtimestamp(value),
                                                                                        inplace=True)
    elif encoder.feature_selection == 'simple':
        dynamic_column = []
        if list(dataset_confs.activity_col.values())[0] == 'Activity':
            dynamic_column.append('prefix')
        elif list(dataset_confs.activity_col.values())[0] == 'Activity Code':
            dynamic_column.append('prefix')
        elif list(dataset_confs.activity_col.values())[0] == 'Activity code':
            dynamic_column.append('prefix')
        long_data = pd.wide_to_long(df, stubnames=dynamic_column, i='Case ID',
                                    j='order', sep='_', suffix=r'\w+')
        timestamps = pd.date_range('1/1/2011', periods=len(long_data), freq='H')

        long_data_sorted = long_data.sort_values(['Case ID', 'order'], ).reset_index(drop=False)
        long_data_sorted[list(dataset_confs.timestamp_col.values())[0]] = timestamps
    elif encoder.feature_selection == 'simple_trace':
        static_columns = list(dataset_confs.case_id_col.values()) + list(dataset_confs.label_col.values()) + list(
            dataset_confs.static_num_cols.values())
        dynamic_column = []
        if list(dataset_confs.activity_col.values())[0] == 'Activity':
            dynamic_column.append('prefix')
        elif list(dataset_confs.activity_col.values())[0] == 'Activity Code':
            dynamic_column.append('prefix')
        elif list(dataset_confs.activity_col.values())[0] == 'Activity code':
            dynamic_column.append('prefix')
        long_data = pd.wide_to_long(df, stubnames=dynamic_column, i='Case ID',
                                    j='order', sep='_', suffix=r'\w+')
        timestamps = pd.date_range('1/1/2011', periods=len(long_data), freq='H')
        long_data_sorted = long_data.sort_values(['Case ID', 'order'], ).reset_index(drop=False)
        long_data_sorted[list(dataset_confs.timestamp_col.values())[0]] = timestamps
    elif encoder.feature_selection == 'loreley':
        if df['prefix'].all() == '0':
            cols = ['prefix_' + str(i + 1) for i in range(CONF['prefix_length'])]
            df[cols] = 0
        else:
            df = pd.concat([df, pd.DataFrame(
                df['prefix'].str.split(",", expand=True).fillna(value='0')).rename(
                columns=lambda x: f"prefix_{int(x) + 1}")], axis=1)
            df = df.replace('\[', '', regex=True)
            df = df.replace(']', '', regex=True)
        df = df.drop(columns=['prefix'])
        static_columns = list(dataset_confs.case_id_col.values()) + list(dataset_confs.label_col.values()) + list(
            dataset_confs.static_num_cols.values())
        dynamic_column = []
        if list(dataset_confs.activity_col.values())[0] == 'Activity':
            dynamic_column.append('prefix')
        elif list(dataset_confs.activity_col.values())[0] == 'Activity Code':
            dynamic_column.append('prefix')
        elif list(dataset_confs.activity_col.values())[0] == 'Activity code':
            dynamic_column.append('prefix')
        long_data = pd.wide_to_long(df, stubnames=dynamic_column, i='Case ID',
                                    j='order', sep='_', suffix=r'\w+')
        timestamps = pd.date_range('1/1/2011', periods=len(long_data), freq='H')

        long_data_sorted = long_data.sort_values(['Case ID', 'order'], ).reset_index(drop=False)
        long_data_sorted[list(dataset_confs.timestamp_col.values())[0]] = timestamps
    long_data_sorted['label'].replace({1: 'regular'}, inplace=True)
    long_data_sorted.drop(columns=['order'], inplace=True)
    columns_to_rename = {'Case ID': 'case:concept:name'}
    columns_to_rename.update({'prefix': 'concept:name'})
    long_data_sorted.rename(columns=columns_to_rename, inplace=True)
    long_data_sorted['label'].replace({'regular': 'false', 'deviant': 'true'}, inplace=True)
    long_data_sorted.replace('0', 'other', inplace=True)
    event_log = pm4py.convert_to_event_log(long_data_sorted)
    d4py.load_xes_log(event_log)
    d4py.compute_frequent_itemsets(min_support=0.95, len_itemset=2, algorithm='apriori')
    discovery = d4py.discovery(consider_vacuity=True, max_declare_cardinality=2)
    filtered_discovery = d4py.filter_discovery(min_support=1,output_path=os.path.join('..','process_models_prefix_length',(dataset+'_'+str(prefix_length)+'.decl')))
    return filtered_discovery

def model_discovery_pipeline(CONF=None):
    logger.info('Hey there!')
    if CONF is None:
        CONF = {  # This contains the configuration for the run
            'data':
                {   'FULL_DATA': '../dataset/' + 'full.xes',
                    'TRAIN_DATA': '../dataset/' + 'train.xes',
                    'VALIDATE_DATA': '../dataset/' + 'validate.xes',
                     'FEEDBACK_DATA': '../dataset/' + 'feedback.xes',
                    'TEST_DATA': '../dataset/' + 'test.xes',
                    'OUTPUT_DATA': '../output_data',
                },
            'prefix_length_strategy': PrefixLengthStrategy.FIXED.value,
            'prefix_length': 15,
            'padding': True,  # TODO, why use of padding?
            'feature_selection': EncodingType.COMPLEX.value,
            'task_generation_type': TaskGenerationType.ONLY_THIS.value,
            'attribute_encoding': EncodingTypeAttribute.LABEL.value,  # LABEL, ONEHOT
            'labeling_type': LabelTypes.ATTRIBUTE_STRING.value,
            'predictive_model': ClassificationMethods.RANDOM_FOREST.value,  # RANDOM_FOREST, LSTM, PERCEPTRON
            'explanator': ExplainerType.DICE.value,  # SHAP, LRP, ICE, DICE
            'threshold': 13,
            'top_k': 10,
            'hyperparameter_optimisation': False,  # TODO, this parameter is not used
            'hyperparameter_optimisation_target': HyperoptTarget.F1.value,
            'hyperparameter_optimisation_epochs': 2,
            'time_encoding': TimeEncodingType.DURATION.value,
            'target_event': None
        }

    logger.debug('LOAD DATA')
    #    full_log = get_log(filepath=CONF['data']['FULL_DATA'])
    train_log = get_log(filepath=CONF['data']['TRAIN_DATA'])
    validate_log = get_log(filepath=CONF['data']['VALIDATE_DATA'])
    test_log = get_log(filepath=CONF['data']['TEST_DATA'])
    feedback_log = get_log(filepath=CONF['data']['FEEDBACK_DATA'])

    logger.debug('ENCODE DATA')

    encoder, train_df = get_encoded_df(
        log=train_log, CONF=CONF, encoder=None, train_cols=None, train_df=None
    )
    _, validate_df = get_encoded_df(
        log=validate_log, CONF=CONF, encoder=encoder, train_cols=train_df.columns, train_df=train_df
    )
    _, test_df = get_encoded_df(
        log=test_log, CONF=CONF, encoder=encoder, train_cols=train_df.columns, train_df=train_df
    )
    _, feedback_df = get_encoded_df(
        log=feedback_log, CONF=CONF, encoder=encoder, train_cols=train_df.columns, train_df=train_df
    )
    full_df = pd.concat([train_df,validate_df,test_df])
    dataset = CONF['data']['TRAIN_DATA'].rpartition('/')[0].replace('../','')
    model_discovery(encoder,full_df,dataset,features_names=full_df.columns[1:-1],prefix_length=CONF['prefix_length'])

if __name__ == '__main__':
    dataset_list = [
    'sepsis_cases_1',
    'sepsis_cases_2',
    'sepsis_cases_4',
    'BPIC15_1_f2',
    'BPIC15_2_f2',
    'BPIC15_3_f2',
    'BPIC15_4_f2',
    'BPIC15_5_f2',
    #'BPIC17_O_ACCEPTED',
    #'BPIC17_O_REFUSED',
    #'BPIC17_O_CANCELLED',
    #'bpic2012_O_ACCEPTED-COMPLETE',
    #'bpic2012_O_CANCELLED-COMPLETE',
    #'bpic2012_O_DECLINED-COMPLETE',
    ]
    #encodings = [EncodingType.COMPLEX.value]
    prefix_lengths = [5,10,15,20,25]
    encodings = [EncodingType.COMPLEX.value]
    for dataset in dataset_list:
        for prefix in prefix_lengths:
            for encoding in encodings:
                CONF = {  # This contains the configuration for the run
                    'data':
                        {#'FULL_DATA':  '../'+dataset+'/' + 'full.xes',
                         'TRAIN_DATA':   '../'+dataset+'/' + 'train.xes',
                         'VALIDATE_DATA':   '../'+dataset+'/' + 'validate.xes',
                         'FEEDBACK_DATA':   '../'+dataset+'/' + 'feedback.xes',
                         'TEST_DATA':   '../'+dataset+'/' + 'test.xes',
                         'OUTPUT_DATA': '../output_data',
                         },
                    'prefix_length_strategy': PrefixLengthStrategy.FIXED.value,
                    'prefix_length': prefix,
                    'padding': True,  # TODO, why use of padding?
                    'feature_selection': encoding,
                    'task_generation_type': TaskGenerationType.ONLY_THIS.value,
                    'attribute_encoding': EncodingTypeAttribute.LABEL.value,  # LABEL, ONEHOT
                    'labeling_type': LabelTypes.ATTRIBUTE_STRING.value,
                    'predictive_model': ClassificationMethods.RANDOM_FOREST.value,  # RANDOM_FOREST, LSTM, PERCEPTRON
                    'explanator': ExplainerType.DICE.value,  # SHAP, LRP, ICE, DICE
                    'threshold': 13,
                    'top_k': 10,
                    'hyperparameter_optimisation': False,  # TODO, this parameter is not used
                    'hyperparameter_optimisation_target': HyperoptTarget.F1.value,
                    'hyperparameter_optimisation_epochs': 2,
                    'time_encoding': TimeEncodingType.NONE.value,
                    'target_event': None,
                }
                model_discovery_pipeline(CONF=CONF)


