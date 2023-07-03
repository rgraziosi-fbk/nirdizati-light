import logging
import numpy as np

import sys
import csv
from src.encoding.common import get_encoded_df, EncodingType
from src.encoding.constants import TaskGenerationType, PrefixLengthStrategy, EncodingTypeAttribute
from src.evaluation.common import evaluate_classifier
from src.explanation.common import ExplainerType, explain
from src.explanation.wrappers.dice_wrapper import plausibility
from src.hyperparameter_optimisation.common import retrieve_best_model, HyperoptTarget
from src.labeling.common import LabelTypes
from src.log.common import get_log
from src.predictive_model.common import ClassificationMethods, get_tensor
from src.predictive_model.predictive_model import PredictiveModel, drop_columns

from src.explanation.visualizations.plot import line_plot,bar_plot
import pandas as pd
import itertools
from datetime import datetime
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def run_simple_pipeline(CONF=None):
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
            'predictive_model': ClassificationMethods.MLP.value,  # RANDOM_FOREST, LSTM, PERCEPTRON
            'explanator': ExplainerType.DICE.value,  # SHAP, LRP, ICE, DICE
            'threshold': 13,
            'top_k': 10,
            'hyperparameter_optimisation': False,  # TODO, this parameter is not used
            'hyperparameter_optimisation_target': HyperoptTarget.AUC.value,
            'hyperparameter_optimisation_epochs': 2,
            'time_encoding': TimeEncodingType.DURATION.value,
            'target_event': None
        }

    logger.debug('LOAD DATA')
    train_log = get_log(filepath=CONF['data']['TRAIN_DATA'])
    validate_log = get_log(filepath=CONF['data']['VALIDATE_DATA'])
    test_log = get_log(filepath=CONF['data']['TEST_DATA'])

    logger.debug('ENCODE DATA')

    encodings = [EncodingType.SIMPLE_TRACE.value]
    for encoding in encodings:
        CONF['feature_selection'] = encoding
        encoder, full_df = get_encoded_df(
            log=train_log, CONF=CONF, encoder=None, train_cols=None, train_df=None
        )
        _, train_df = get_encoded_df(
            log=train_log, CONF=CONF, encoder=encoder, train_cols=full_df.columns, train_df=full_df
        )

        _, validate_df = get_encoded_df(
            log=validate_log, CONF=CONF, encoder=encoder, train_cols=full_df.columns, train_df=full_df
        )
        _, test_df = get_encoded_df(
            log=test_log, CONF=CONF, encoder=encoder, train_cols=full_df.columns, train_df=full_df
        )
        logger.debug('TRAIN PREDICTIVE MODEL')
        # change label values
        def change_df_label(df):
            df.iloc[:,-1] = df.iloc[:,-1] - 1
            return df
        #full_df = change_df_label(full_df)
        train_df = change_df_label(train_df)
        test_df = change_df_label(test_df)
        validate_df = change_df_label(validate_df)

        predictive_model = PredictiveModel(CONF, CONF['predictive_model'], train_df, validate_df)
        predictive_model.model, predictive_model.config = retrieve_best_model(
            predictive_model,
            CONF['predictive_model'],
            max_evaluations=CONF['hyperparameter_optimisation_epochs'],
            target=CONF['hyperparameter_optimisation_target']
        )

        logger.debug('EVALUATE PREDICTIVE MODEL')
        if predictive_model.model_type is ClassificationMethods.LSTM.value:
            probabilities = predictive_model.model.predict(get_tensor(CONF, drop_columns(test_df)))
            predicted = np.argmax(probabilities, axis=1)
            scores = np.amax(probabilities, axis=1)
        #elif predictive_model.model_type is ClassificationMethods.MLP.value:
        #    probabilities = predictive_model.model.predict(drop_columns(test_df))
        #    predicted = [1 if prob > 0.5 else 0 for prob in probabilities]
        #    scores = np.amax(probabilities, axis=1)
        elif predictive_model.model_type not in (ClassificationMethods.LSTM.value):
            predicted = predictive_model.model.predict(drop_columns(test_df))
            scores = predictive_model.model.predict_proba(drop_columns(test_df))[:, 1]

        actual = test_df['label']
        if predictive_model.model_type is ClassificationMethods.LSTM.value:
            actual = np.array(actual.to_list())

        initial_result = evaluate_classifier(actual, predicted, scores)
        logger.debug('COMPUTE EXPLANATION')
        if CONF['explanator'] == ExplainerType.DICE.value:
            #list = ['gender','sex','race','ethnicity']
            list = []
            sensitive_features = [column for column in train_df.columns if any(feature in column for feature in list)]
            features_to_vary = [col for col in predictive_model.train_df.columns if col not in sensitive_features]
            #set test df just with correctly predicted labels
            test_df_correct = test_df[test_df['label']==predicted]
            test_df_correct = test_df[test_df['label'] == 0]
            cf_dataset = pd.concat([train_df,validate_df],ignore_index=True)
            full_df = pd.concat([train_df,validate_df,test_df])
            cf_dataset.loc[len(cf_dataset)] = 0
            methods = ['genetic_conformance']
            optimizations = ['loss_function','filtering']
            heuristics = ['heuristic_2']
            for method in methods:
                for heuristic in heuristics:
                    for optimization in optimizations:
                        explanations = explain(CONF, predictive_model,encoder=encoder,cf_df=full_df.iloc[:,1:],
                                           query_instances=test_df_correct.iloc[:,1:],
                                           features_to_vary=features_to_vary,method=method,df=full_df.iloc[:,1:],optimization=optimization,
                                           heuristic=heuristic)
        del encoder, train_df, test_df, validate_df
    logger.info('RESULT')
    logger.info('INITIAL', initial_result)
    logger.info('Done, cheers!')


    #return { 'initial_result', initial_result, 'predictive_model.config', predictive_model.config}
if __name__ == '__main__':
    dataset_list = [
    #'hospital_billing_2',
    #'hospital_billing_3'
    #'synthetic_data',
    #'BPIC11_f1',
    'bpic2012_O_ACCEPTED-COMPLETE',
    #'BPIC15_1_f2',
    #'BPIC15_2_f2',
    #'BPIC15_3_f2',
    #'BPIC15_4_f2',
    #'BPIC15_5_f2',
    #'bpic2012_O_DECLINED-COMPLETE',
    #'bpic2012_O_CANCELLED-COMPLETE',
    #'traffic_fines_1',
    #'synthetic_data'
    #'Production'
    ]
    prefix_lengths = [0.2]
    #prefix_lengths = [0.6]
    for dataset in dataset_list:
        for prefix in prefix_lengths:
            CONF = {  # This contains the configuration for the run
                'data':
                    {'FULL_DATA':  '../'+dataset+'/' + 'full.xes',
                     'TRAIN_DATA':   '../'+dataset+'/' + 'train.xes',
                     'VALIDATE_DATA':   '../'+dataset+'/' + 'validate.xes',
                     'FEEDBACK_DATA':  '../'+dataset+'/' + 'feedback.xes',
                     'TEST_DATA':   '../'+dataset+'/' + 'test.xes',
                     'OUTPUT_DATA': '../output_data',
                     },
                'prefix_length_strategy': PrefixLengthStrategy.PERCENTAGE.value,
                'prefix_length':prefix,
                'padding': True,  # TODO, why use of padding?
                'feature_selection': None,
                'task_generation_type': TaskGenerationType.ONLY_THIS.value,
                'attribute_encoding': EncodingTypeAttribute.LABEL.value,  # LABEL, ONEHOT
                'labeling_type': LabelTypes.ATTRIBUTE_STRING.value,
                'predictive_model': ClassificationMethods.RANDOM_FOREST.value,  # RANDOM_FOREST, LSTM, PERCEPTRON
                'explanator': ExplainerType.DICE.value,  # SHAP, LRP, ICE, DICE
                'threshold': 13,
                'top_k': 10,
                'hyperparameter_optimisation': False,  # TODO, this parameter is not used
                'hyperparameter_optimisation_target': HyperoptTarget.F1.value,
                'hyperparameter_optimisation_epochs': 10,
                'time_encoding': TimeEncodingType.NONE.value,
                'target_event': None,
            }
            run_simple_pipeline(CONF=CONF)

