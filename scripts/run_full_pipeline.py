import itertools
import logging
import numpy as np

from nirdizati_light.encoding.common import get_encoded_df, EncodingType
from nirdizati_light.encoding.constants import EncodingTypeAttribute
from nirdizati_light.evaluation.common import evaluate_classifier
from nirdizati_light.explanation.common import explain, ExplainerType
from nirdizati_light.confusion_matrix_feedback.confusion_matrix_feedback import compute_feedback
from nirdizati_light.confusion_matrix_feedback.randomise_features import randomise_features
from nirdizati_light.hyperparameter_optimisation.common import retrieve_best_model, HyperoptTarget
from nirdizati_light.labeling.common import LabelTypes
from nirdizati_light.log.common import get_log
from nirdizati_light.predictive_model.common import ClassificationMethods, get_tensor
from nirdizati_light.predictive_model.predictive_model import PredictiveModel, drop_columns

logger = logging.getLogger(__name__)


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def run_full_pipeline(CONF=None):
    logger.info('Hey there!')
    if CONF is None:
        CONF = {  # This contains the configuration for the run
            'data':
                {
                    'TRAIN_DATA': '../synthetic_data/' + 'd1_train_explainability_0-38.xes',
                    'VALIDATE_DATA': '../synthetic_data/' + 'validate.xes',
                    'FEEDBACK_DATA': '../synthetic_data/' + 'feedback.xes',
                    'TEST_DATA': '../synthetic_data/' + 'test.xes',
                    'OUTPUT_DATA': '../output_data',
                },
            'prefix_length': 5,
            'padding': True,  # TODO: why use of padding?
            'feature_selection': EncodingType.SIMPLE.value,
            'attribute_encoding': EncodingTypeAttribute.ONEHOT.value,  # LABEL, ONEHOT
            'labeling_type': LabelTypes.ATTRIBUTE_STRING.value,
            'predictive_model': ClassificationMethods.RANDOM_FOREST.value,  # RANDOM_FOREST, LSTM
            'explanator': ExplainerType.SHAP.value,  # SHAP, LRP
            'threshold': 13,
            'top_k': 10,
            'hyperparameter_optimisation': False,  # TODO: this parameter is not used
            'hyperparameter_optimisation_target': HyperoptTarget.AUC.value,
            'hyperparameter_optimisation_epochs': 20,  # 100 TODO set a higher value
        }

    logger.debug('LOAD DATA')
    full_log = get(filepath=CONF['data']['FULL_DATA'])
    train_log = get_log(filepath=CONF['data']['TRAIN_DATA'])
    validate_log = get_log(filepath=CONF['data']['VALIDATE_DATA'])
    feedback_log = get_log(filepath=CONF['data']['FEEDBACK_DATA'])
    test_log = get_log(filepath=CONF['data']['TEST_DATA'])

    logger.debug('ENCODE DATA')

    encoder, full_df = get_encoded_df(
        log=full_log,
        CONF=CONF
    )
    encoder, train_df = get_encoded_df(
        log=train_log,
        CONF=CONF,
        encoder=encoder
    )
    encoder, validate_df = get_encoded_df(
        log=validate_log,
        encoder=encoder,
        CONF=CONF
    )
    encoder, feedback_df = get_encoded_df(
        log=feedback_log,
        encoder=encoder,
        CONF=CONF
    )
    encoder, test_df = get_encoded_df(
        log=test_log,
        encoder=encoder,
        CONF=CONF
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

    logger.debug('COMPUTE EXPLANATION')
    explanations = explain(CONF, predictive_model, feedback_df, encoder)

    logger.debug('COMPUTE FEEDBACK')
    feedback_10 = compute_feedback(
        CONF,
        explanations,
        predictive_model,
        feedback_df,
        encoder,
        top_k=CONF['top_k']
    )

    logger.debug('SHUFFLE FEATURES')
    encoder.decode(train_df)
    encoder.decode(validate_df)
    returned_results = {}
    for top_k_threshold in [1, 2]:
        feedback = {classes: feedback_10[classes][:top_k_threshold] for classes in feedback_10}

        retrain_results = []
        for _ in range(2):

            shuffled_train_df = randomise_features(feedback, train_df)
            shuffled_validate_df = randomise_features(feedback, validate_df)
            encoder.encode(shuffled_train_df)
            encoder.encode(shuffled_validate_df)

            logger.debug('RETRAIN-- TRAIN PREDICTIVE MODEL')
            predictive_model = PredictiveModel(CONF, CONF['predictive_model'], shuffled_train_df, shuffled_validate_df)
            try:
                predictive_model.model, predictive_model.config = retrieve_best_model(
                    predictive_model,
                    CONF['predictive_model'],
                    max_evaluations=CONF['hyperparameter_optimisation_epochs'],
                    target=CONF['hyperparameter_optimisation_target']
                )

                logger.debug('RETRAIN-- EVALUATE PREDICTIVE MODEL')
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

                retrain_results += [evaluate_classifier(actual, predicted, scores)]
            except Exception as e:
                pass

        returned_results[top_k_threshold] = {
            'avg': dict_mean(retrain_results),
            'retrain_results': retrain_results
        }

    logger.info('RESULT')
    logger.info('INITIAL', initial_result)
    logger.info('RETRAIN', returned_results)

    logger.info('Done, cheers!')

    return {'feedback_10': feedback_10, 'used_feedback': feedback, 'initial_result': initial_result, 'retrain_result': returned_results}


if __name__ == '__main__':
    run_full_pipeline()

