import itertools
import logging
import numpy as np

from src.encoding.common import get_encoded_df, EncodingType
from src.encoding.constants import TaskGenerationType, PrefixLengthStrategy, EncodingTypeAttribute
from src.evaluation.common import evaluate
from src.explanation.common import explain, ExplainerType
from src.confusion_matrix_feedback.confusion_matrix_feedback import compute_feedback
from src.confusion_matrix_feedback.randomise_features import randomise_features
from src.hyperparameter_optimisation.common import retrieve_best_model, HyperoptTarget
from src.labeling.common import LabelTypes
from src.log.common import get_log
from src.predictive_model.common import PredictionMethods, get_tensor
from src.predictive_model.predictive_model import PredictiveModel, drop_columns
from src.encoding.time_encoding import TimeEncodingType

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
                    'FEEDBACK_DATA': '../input_data/' + 'd1_test_explainability_40-50.xes',
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
            'predictive_model': PredictionMethods.RANDOM_FOREST.value,  # RANDOM_FOREST, LSTM
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

    logger.debug('TRAIN PREDICTIVE MODEL')
    predictive_model = PredictiveModel(CONF, CONF['predictive_model'], train_df, validate_df)

    predictive_model.model, predictive_model.config = retrieve_best_model(
        predictive_model,
        CONF['predictive_model'],
        max_evaluations=CONF['hyperparameter_optimisation_epochs'],
        target=CONF['hyperparameter_optimisation_target']
    )

    logger.debug('EVALUATE PREDICTIVE MODEL')
    if predictive_model.model_type is not PredictionMethods.LSTM.value:
        predicted = predictive_model.model.predict(drop_columns(test_df))
        scores = predictive_model.model.predict_proba(drop_columns(test_df))[:, 1]
    elif predictive_model.model_type is PredictionMethods.LSTM.value:
        probabilities = predictive_model.model.predict(get_tensor(CONF, drop_columns(test_df)))
        predicted = np.argmax(probabilities, axis=1)
        scores = np.amax(probabilities, axis=1)

    actual = test_df['label']
    if predictive_model.model_type is PredictionMethods.LSTM.value:
        actual = np.argmax(np.array(actual.to_list()), axis=1)

    initial_result = evaluate(actual, predicted, scores)

    logger.info('RESULT')
    logger.info('INITIAL', initial_result)

    logger.info('Done, cheers!')

    return { 'initial_result': initial_result, 'predictive_model.config': predictive_model.config}


if __name__ == '__main__':
    dic = run_simple_pipeline()
    print(str(dic))


