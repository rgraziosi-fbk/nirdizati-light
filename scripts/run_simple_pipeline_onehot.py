import logging
import numpy as np
import pandas as pd

from src.encoding.common import get_encoded_df, EncodingType
from src.encoding.constants import TaskGenerationType, PrefixLengthStrategy, EncodingTypeAttribute
from src.encoding.time_encoding import TimeEncodingType
from src.evaluation.common import evaluate_classifier
from src.explanation.common import ExplainerType
from src.hyperparameter_optimisation.common import retrieve_best_model, HyperoptTarget
from src.labeling.common import LabelTypes
from src.log.common import get_log
from src.predictive_model.common import ClassificationMethods, get_tensor
from src.predictive_model.predictive_model import PredictiveModel, drop_columns

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
                {
                    'TRAIN_DATA': '../input_data/' + 'd1_train_explainability_0-38.xes',
                    'VALIDATE_DATA': '../input_data/' + 'd1_validation_explainability_38-40.xes',
                    # 'FEEDBACK_DATA': '../input_data/' + 'd1_test_explainability_40-50.xes',
                    'TEST_DATA': '../input_data/' + 'd1_test2_explainability_50-60.xes',
                    'OUTPUT_DATA': '../output_data',
                },
            'prefix_length_strategy': PrefixLengthStrategy.PERCENTAGE.value,
            'prefix_length': 0.3,
            'padding': True,  # TODO: why use of padding?
            'feature_selection': EncodingType.SIMPLE.value,
            'task_generation_type': TaskGenerationType.ALL_IN_ONE.value,
            'attribute_encoding': EncodingTypeAttribute.LABEL.value,  # LABEL, ONEHOT
            'labeling_type': LabelTypes.NEXT_ACTIVITY.value,
            'predictive_model': ClassificationMethods.RANDOM_FOREST.value,  # RANDOM_FOREST, LSTM
            'explanator': ExplainerType.SHAP.value,  # SHAP, LRP
            'threshold': 13,
            'top_k': 10,
            'hyperparameter_optimisation': False,  # TODO: this parameter is not used
            'hyperparameter_optimisation_target': HyperoptTarget.F1.value,
            'hyperparameter_optimisation_epochs': 10,  # 100 TODO set a higher value
        }

    logger.debug('LOAD DATA')
    train_log = get_log(filepath=CONF['data']['TRAIN_DATA'])
    validate_log = get_log(filepath=CONF['data']['VALIDATE_DATA'])
    test_log = get_log(filepath=CONF['data']['TEST_DATA'])

    logger.debug('ENCODE DATA')
    encoder, train_df = get_encoded_df(
        log=train_log,
        CONF=CONF
    )
    encoder, validate_df = get_encoded_df(
        log=validate_log,
        encoder=encoder,
        CONF=CONF,
        train_df=train_df
    )
    encoder, test_df = get_encoded_df(
        log=test_log,
        encoder=encoder,
        CONF=CONF,
        train_df=train_df
    )

    def one_hot_encode(df, encoder):
        columns = ['trace_id'] + [
            feature_name + '//' + feature_value
            for feature_name in encoder._label_dict
            for feature_value in encoder._label_dict[feature_name]
        ] + ['label']
        data = [
            [row.loc['trace_id']] +
            [False] * (len(columns) - 2) +
            [row.loc['label']]
            for index, row in df.iterrows()
        ]
        return_df = pd.DataFrame(columns=columns, data=data)
        encoder.decode(df)
        for index, row in df.iterrows():
            for feature_name, feature_value in row.iteritems():
                if feature_name not in ['trace_id', 'label']:
                    return_df.at[index, feature_name + '//' + str(feature_value)] = True
        encoder.encode(df)
        return return_df

    one_hot_encoded_train_df = one_hot_encode(train_df, encoder)
    one_hot_encoded_validate_df = one_hot_encode(validate_df, encoder)
    one_hot_encoded_test_df = one_hot_encode(test_df, encoder)

    logger.debug('TRAIN PREDICTIVE MODEL')
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
        actual = np.argmax(np.array(actual.to_list()), axis=1)

    initial_result = evaluate_classifier(actual, predicted, scores)

    CONF2 = CONF
    CONF2['task_generation_type'] = TaskGenerationType.ONLY_THIS.value
    encoder, test_df = get_encoded_df(
        log=test_log,
        encoder=encoder,
        CONF=CONF2,
        train_df=train_df
    )
    predicted = predictive_model.model.predict(drop_columns(test_df))
    scores = predictive_model.model.predict_proba(drop_columns(test_df))[:, 1]
    results2 = evaluate_classifier(actual, predicted, scores)

    logger.info('RESULT')
    logger.info('INITIAL', initial_result)

    logger.info('Done, cheers!')

    return { 'initial_result': initial_result, 'predictive_model.config': predictive_model.config}


if __name__ == '__main__':
    dic = run_simple_pipeline({  # This contains the configuration for the run
            'data':
                {
                    'TRAIN_DATA': '../input_data/BPI11_f2/train.xes',
                    'VALIDATE_DATA': '../input_data/BPI11_f2/validate.xes',
                    'TEST_DATA': '../input_data/BPI11_f2/f2_80-100.xes',
                    'OUTPUT_DATA': '../output_data',
                },
            'prefix_length_strategy': PrefixLengthStrategy.FIXED.value,
            'prefix_length': 7,
            'padding': True,  # TODO: why use of padding?
            'feature_selection': EncodingType.COMPLEX.value,
            'task_generation_type': TaskGenerationType.ALL_IN_ONE.value,
            'attribute_encoding': EncodingTypeAttribute.LABEL.value,  # LABEL, ONEHOT
            'labeling_type': LabelTypes.ATTRIBUTE_STRING.value,
            'predictive_model': ClassificationMethods.RANDOM_FOREST.value,  # RANDOM_FOREST, LSTM
            'explanator': ExplainerType.SHAP.value,  # SHAP, LRP
            'threshold': 13,
            'top_k': 10,
            'hyperparameter_optimisation': True,  # TODO: this parameter is not used
            'hyperparameter_optimisation_target': HyperoptTarget.AUC.value,
            'hyperparameter_optimisation_epochs': 10,  # 100 TODO set a higher value
            'target_event': None,
            'time_encoding': TimeEncodingType.NONE.value
        })
    print(str(dic))


