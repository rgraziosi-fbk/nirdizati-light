import logging
import numpy as np
import sys
from src.encoding.common import get_encoded_df, EncodingType
from src.encoding.constants import TaskGenerationType, PrefixLengthStrategy, EncodingTypeAttribute
from src.encoding.time_encoding import TimeEncodingType
from src.evaluation.common import evaluate_classifier
from src.explanation.common import ExplainerType, explain
from src.hyperparameter_optimisation.common import retrieve_best_model, HyperoptTarget
from src.labeling.common import LabelTypes
from src.log.common import get_log
from src.predictive_model.common import ClassificationMethods, get_tensor
from src.predictive_model.predictive_model import PredictiveModel, drop_columns
from src.explanation.visualizations.plot import line_plot,bar_plot
import dice_ml
import pandas as pd
from dice_ml.utils import helpers
logger = logging.getLogger(__name__)


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
                {   'FULL_DATA': '../dataset/' + '100.xes',
                    'TRAIN_DATA': '../dataset/' + 'train.xes',
                    'VALIDATE_DATA': '../dataset/' + 'validate.xes',
                     'FEEDBACK_DATA': '../dataset/' + 'feedback.xes',
                    'TEST_DATA': '../dataset/' + 'test.xes',
                    'OUTPUT_DATA': '../output_data',
                },
            'prefix_length_strategy': PrefixLengthStrategy.FIXED.value,
            'prefix_length': 3,
            'padding': True,  # TODO: why use of padding?
            'feature_selection': EncodingType.COMPLEX.value,
            'task_generation_type': TaskGenerationType.ONLY_THIS.value,
            'attribute_encoding': EncodingTypeAttribute.LABEL.value,  # LABEL, ONEHOT
            'labeling_type': LabelTypes.ATTRIBUTE_STRING.value,
            'predictive_model': ClassificationMethods.LSTM.value,  # RANDOM_FOREST, LSTM, PERCEPTRON
            'explanator': ExplainerType.DICE.value,  # SHAP, LRP, ICE, DICE
            'threshold': 13,
            'top_k': 10,
            'hyperparameter_optimisation': False,  # TODO: this parameter is not used
            'hyperparameter_optimisation_target': HyperoptTarget.F1.value,
            'hyperparameter_optimisation_epochs': 2,
            'time_encoding': TimeEncodingType.NONE.value,
            'target_event': None
        }

    logger.debug('LOAD DATA')
    full_log = get_log(filepath=CONF['data']['FULL_DATA'])
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
    logger.debug('TRAIN PREDICTIVE MODEL')
    # change label values

    def change_df_label(df):
        df.iloc[:,-1] = df.iloc[:,-1] - 1
        return df
    #full_df = change_df_label(full_df)
    train_df = change_df_label(train_df)
    test_df = change_df_label(test_df)
    validate_df = change_df_label(validate_df)
    feedback_df = change_df_label(feedback_df)

    predictive_model = PredictiveModel(CONF, CONF['predictive_model'], train_df, validate_df)

    predictive_model.model, predictive_model.config = retrieve_best_model(
        predictive_model,
        CONF['predictive_model'],
        max_evaluations=CONF['hyperparameter_optimisation_epochs'],
        target=CONF['hyperparameter_optimisation_target']
    )
    logger.debug('EVALUATE PREDICTIVE MODEL')
    if predictive_model.model_type is not ClassificationMethods.LSTM.value:
        predicted = predictive_model.model.predict(drop_columns(test_df))
        scores = predictive_model.model.predict_proba(drop_columns(test_df))[:, 1]
    elif predictive_model.model_type is ClassificationMethods.LSTM.value:
        probabilities = predictive_model.model.predict(get_tensor(CONF, drop_columns(test_df)))
        predicted = np.argmax(probabilities, axis=1)
        scores = np.amax(probabilities, axis=1)

    actual = test_df['label']
    if predictive_model.model_type is ClassificationMethods.LSTM.value:
        actual = np.array(actual.to_list())

    initial_result = evaluate_classifier(actual, predicted, scores)
    logger.debug('COMPUTE EXPLANATION')
    list = ['Age','gender','sex','race','ethnicity']
    sensitive_features = [column for column in train_df.columns if any(feature in column for feature in list)]
    features_to_vary = [col for col in predictive_model.train_df.columns if col not in sensitive_features]
    # For unknown
    cf_dataset = pd.concat([train_df,validate_df],ignore_index=True)
    cf_dataset.loc[len(cf_dataset)] = 0
    #change full_df here because you do not want to use the whole data, or you could
    explanations = explain(CONF, predictive_model,encoder=encoder,cf_df=cf_dataset.iloc[:,1:],
                           query_instances=test_df.iloc[:,1:],
                           features_to_vary=features_to_vary,method='random'
                           )

    logger.info('RESULT')
    logger.info('INITIAL', initial_result)

    logger.info('Done, cheers!')

    #return { 'initial_result': initial_result, 'predictive_model.config': predictive_model.config}


if __name__ == '__main__':
    dataset_list = ['bpic2012_O_ACCEPTED-COMPLETE', 'bpic2012_O_CANCELLED-COMPLETE', 'bpic2012_O_DECLINED-COMPLETE',
                    'BPIC15_1_f2', 'BPIC15_2_f2', 'BPIC15_3_f2', 'BPIC15_4_f2', 'BPIC15_5_f2',
                    'Production', 'BPIC11_f4', 'BPIC11_f2', 'BPIC11_f1',
                    'sepsis_cases_1', 'sepsis_cases_2', 'sepsis_cases_4']



    for dataset in dataset_list:
        CONF = {  # This contains the configuration for the run
            'data':
                {'FULL_DATA':  dataset+'/' + '100.xes',
                 'TRAIN_DATA':  dataset+'/' + 'train.xes',
                 'VALIDATE_DATA':  dataset+'/' + 'validate.xes',
                 'FEEDBACK_DATA':  dataset+'/' + 'feedback.xes',
                 'TEST_DATA':  dataset+'/' +  'test.xes',
                 'OUTPUT_DATA': '../output_data',
                 },
            'prefix_length_strategy': PrefixLengthStrategy.FIXED.value,
            'prefix_length': 3,
            'padding': True,  # TODO: why use of padding?
            'feature_selection': EncodingType.COMPLEX.value,
            'task_generation_type': TaskGenerationType.ONLY_THIS.value,
            'attribute_encoding': EncodingTypeAttribute.ONEHOT.value,  # LABEL, ONEHOT
            'labeling_type': LabelTypes.ATTRIBUTE_STRING.value,
            'predictive_model': ClassificationMethods.LSTM.value,  # RANDOM_FOREST, LSTM, PERCEPTRON
            'explanator': ExplainerType.DICE.value,  # SHAP, LRP, ICE, DICE
            'threshold': 13,
            'top_k': 10,
            'hyperparameter_optimisation': False,  # TODO: this parameter is not used
            'hyperparameter_optimisation_target': HyperoptTarget.F1.value,
            'hyperparameter_optimisation_epochs': 2,
            'time_encoding': TimeEncodingType.NONE.value,
            'target_event': None
        }
        run_simple_pipeline(CONF=CONF)

