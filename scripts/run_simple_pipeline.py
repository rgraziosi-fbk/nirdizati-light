import logging
import warnings
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nirdizati_light.encoding.common import get_encoded_df, EncodingType
from nirdizati_light.encoding.constants import TaskGenerationType, PrefixLengthStrategy, EncodingTypeAttribute
from nirdizati_light.encoding.time_encoding import TimeEncodingType
from nirdizati_light.evaluation.common import evaluate_classifier
from nirdizati_light.explanation.common import ExplainerType, explain
from nirdizati_light.hyperparameter_optimisation.common import retrieve_best_model, HyperoptTarget
from nirdizati_light.labeling.common import LabelTypes
from nirdizati_light.log.common import get_log
from nirdizati_light.predictive_model.common import ClassificationMethods, get_tensor
from nirdizati_light.predictive_model.predictive_model import PredictiveModel, drop_columns

from dataset_confs import DatasetConfs

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def run_simple_pipeline(CONF=None, dataset_name=None):
    dataset_confs = DatasetConfs(dataset_name=dataset_name, where_is_the_file=CONF['data'])

    logger.debug('LOAD DATA')
    log = get_log(filepath=CONF['data'])

    logger.debug('ENCODE DATA')
    encodings = [EncodingType.SIMPLE.value, EncodingType.SIMPLE_TRACE.value, EncodingType.COMPLEX.value,
                 EncodingType.LORELEY.value]
    for encoding in encodings:
        CONF['feature_selection'] = encoding
        encoder, full_df = get_encoded_df(log=log, CONF=CONF)

        logger.debug('TRAIN PREDICTIVE MODEL')

        # change label values
        full_df.iloc[:, -1] -= 1

        # split in train, val, test
        train_size = CONF['train_val_test_split'][0]
        val_size = CONF['train_val_test_split'][1]
        test_size = CONF['train_val_test_split'][2]
        if train_size + val_size + test_size != 1.0:
            raise Exception('Train-val-test split does not sum up to 1')

        #Train test split, works by splitting first in train_size (e.g. 60%), then the remaining 40% in val_size and test_size
        #70% - 15% - 15% is the default
        train_df,val_df,test_df = np.split(full_df,[int(train_size*len(full_df)), int((train_size+val_size)*len(full_df))])


        predictive_model = PredictiveModel(CONF, CONF['predictive_model'], train_df, val_df)
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
        elif predictive_model.model_type not in (ClassificationMethods.LSTM.value):
            predicted = predictive_model.model.predict(drop_columns(test_df))
            scores = predictive_model.model.predict_proba(drop_columns(test_df))[:, 1]

        actual = test_df['label']
        if predictive_model.model_type is ClassificationMethods.LSTM.value:
            actual = np.array(actual.to_list())

        initial_result = evaluate_classifier(actual, predicted, scores)

        logger.debug('COMPUTE EXPLANATION')
        if CONF['explanator'] == ExplainerType.DICE.value:
            # list = ['gender','sex','race','ethnicity']
            list = []
            sensitive_features = [column for column in train_df.columns if any(feature in column for feature in list)]
            features_to_vary = [col for col in predictive_model.train_df.columns if col not in sensitive_features]
            # set test df just with correctly predicted labels
            test_df_correct = test_df[(test_df['label'] == predicted) & (test_df['label'] == 0)]
            cf_dataset = pd.concat([train_df, val_df], ignore_index=True)
            full_df = pd.concat([train_df, val_df, test_df])
            cf_dataset.loc[len(cf_dataset)] = 0
            methods = ['genetic_conformance']
            optimizations = ['filtering', 'loss_function']
            heuristics = ['heuristic_2']
            model_path = '../experiments/process_models/process_models'
            for method in methods:
                for heuristic in heuristics:
                    for optimization in optimizations:
                        explanations = explain(CONF, predictive_model, encoder=encoder, cf_df=full_df.iloc[:, 1:],
                                               query_instances=test_df_correct.iloc[:, 1:],
                                               features_to_vary=features_to_vary,
                                               method=method, df=full_df.iloc[:, 1:], optimization=optimization,
                                               heuristic=heuristic, support=0.8,
                                               timestamp_col_name=[*dataset_confs.timestamp_col.values()][0],
                                               model_path=model_path)

        del encoder, train_df, test_df, validate_df
    logger.info('RESULT')
    logger.info('INITIAL', initial_result)
    logger.info('Done, cheers!')

    # return { 'initial_result', initial_result, 'predictive_model.config', predictive_model.config}


if __name__ == '__main__':
    dataset_list = [
        # 'hospital_billing_2',
        # 'hospital_billing_3'
        # 'synthetic_data',
        # 'bpic2012_O_ACCEPTED-COMPLETE',
        # 'BPIC15_1_f2',
        # 'BPIC15_2_f2',
        # 'BPIC15_3_f2',
        # 'BPIC15_4_f2',
        # 'BPIC15_5_f2',
        # 'bpic2012_O_DECLINED-COMPLETE',
        # 'bpic2012_O_CANCELLED-COMPLETE',
        # 'traffic_fines_1',
        'sepsis_cases_1',
        'legal_complaints'
    ]
    prefix_lengths = [0.2, 0.4, 0.6, 0.8, 1]

    for dataset in dataset_list:
        for prefix in prefix_lengths:
            CONF = {  # This contains the configuration for the run
                'data': os.path.join('..','datasets',dataset, 'full.xes'),
                'train_val_test_split': [0.7, 0.15, 0.15],
                'output': os.path.join('..', 'output_data'),
                'prefix_length_strategy': PrefixLengthStrategy.PERCENTAGE.value,
                'prefix_length': prefix,
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
                'seed': 42,
            }

            run_simple_pipeline(CONF=CONF, dataset_name=dataset)
