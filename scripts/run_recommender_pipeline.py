import logging

from src.encoding.common import get_encoded_df, EncodingType
from src.encoding.constants import TaskGenerationType, PrefixLengthStrategy, EncodingTypeAttribute
from src.encoding.time_encoding import TimeEncodingType
from src.evaluation.common import evaluate_recommender
from src.hyperparameter_optimisation.common import retrieve_best_model, HyperoptTarget
from src.labeling.common import LabelTypes
from src.log.common import get_log
from src.predictive_model.common import ClassificationMethods, RegressionMethods
from src.predictive_model.predictive_model import PredictiveModel, drop_columns
from src.recommender_model.classifier_and_regressor.classifier_and_regressor import RecommenderModelInstantiation
from src.recommender_model.common import RecommendationMethods
from src.recommender_model.recommender_model import RecommenderModel

logger = logging.getLogger(__name__)


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def run_recommender_pipeline(CONF=None):
    logger.info('Hey there!')
    if CONF is None:
        CONF = {  # This contains the configuration for the run
            'data':
                {
                    'TRAIN_DATA': '../input_data/' + 'd1_train_explainability_0-38.xes',
                    'VALIDATE_DATA': '../input_data/' + 'd1_validation_explainability_38-40.xes',
                    'TEST_DATA': '../input_data/' + 'd1_test2_explainability_50-60.xes',
                    'OUTPUT_DATA': '../output_data',
                },
            'prefix_length_strategy': PrefixLengthStrategy.FIXED.value,
            'prefix_length': 4, # THIS IS THE ACTUAL VALUE, THE classifier is built with this, the regressor with next
            'padding': True,  # TODO: why use of padding?
            'feature_selection': EncodingType.SIMPLE.value,
            'task_generation_type': TaskGenerationType.ALL_IN_ONE.value,
            'attribute_encoding': EncodingTypeAttribute.LABEL.value,  # LABEL, ONEHOT
            'labeling_type': LabelTypes.NEXT_ACTIVITY.value,
            'recommender_model': RecommendationMethods.CLASSIFICATION_AND_REGRESSION.value,
            'hyperparameter_optimisation': False,  # TODO: this parameter is not used
            'hyperparameter_optimisation_target': {
                RecommenderModelInstantiation.CLASSIFIER.value: HyperoptTarget.F1.value,
                RecommenderModelInstantiation.REGRESSOR.value: HyperoptTarget.MAE.value},
            'hyperparameter_optimisation_epochs': 3,  # 100 TODO set a higher value
            'time_encoding': TimeEncodingType.NONE.value,
            'target_event': None
        }

    logger.debug('LOAD DATA')
    train_log = get_log(filepath=CONF['data']['TRAIN_DATA'])
    validate_log = get_log(filepath=CONF['data']['VALIDATE_DATA'])
    test_log = get_log(filepath=CONF['data']['TEST_DATA'])

    logger.debug('ENCODE DATA-- CLASSIFIER')
    CONF_CLASSIFIER = dict(CONF)
    CONF_CLASSIFIER['predictive_model'] = ClassificationMethods.RANDOM_FOREST.value
    classifier_encoder, classifier_train_df = get_encoded_df(
        log=train_log,
        CONF=CONF_CLASSIFIER
    )
    classifier_encoder, classifier_validate_df = get_encoded_df(
        log=validate_log,
        encoder=classifier_encoder,
        CONF=CONF_CLASSIFIER,
        train_df=classifier_train_df
    )

    logger.debug('ENCODE DATA-- REGRESSOR')
    CONF_REGRESSOR = dict(CONF)
    CONF_REGRESSOR['prefix_length'] += 1
    CONF_REGRESSOR['predictive_model'] = RegressionMethods.RANDOM_FOREST.value

    def label_duration(log, CONF):
        for trace in log:
            trace.attributes['label'] = None
            # time between current prefix and end of trace
            trace.attributes['label'] = \
                trace[-1]['time:timestamp'] - \
                trace[CONF['prefix_length']]['time:timestamp'] #time is exact, maybe better tassellating it
            trace.attributes['label'] = trace.attributes['label'].seconds
        return log

    relabeled_train_log = label_duration(train_log, CONF_REGRESSOR)
    relabeled_validate_log = label_duration(validate_log, CONF_REGRESSOR)

    encoder, regressor_train_df = get_encoded_df(
        log=relabeled_train_log,
        CONF=CONF
    )
    encoder, regressor_validate_df = get_encoded_df(
        log=relabeled_validate_log,
        encoder=encoder,
        CONF=CONF,
        train_df=regressor_train_df
    )

    logger.debug('TRAIN RECOMMENDER MODEL')
    recommender_model = RecommenderModel(
        CONF,
        CONF_CLASSIFIER, classifier_train_df, classifier_validate_df,
        CONF_REGRESSOR, regressor_train_df, regressor_validate_df
    )
    recommender_model.retrieve_best_model(
        CONF['hyperparameter_optimisation_epochs'],
        target=CONF['hyperparameter_optimisation_target']
    )

    logger.debug('EVALUATE PREDICTIVE MODEL')
    CONF_TEST = dict(CONF_CLASSIFIER)
    CONF_TEST['task_generation_type'] = TaskGenerationType.ONLY_THIS.value
    CONF_TEST['prefix_length_strategy'] = PrefixLengthStrategy.FIXED.value
    CONF_TEST['prefix_length'] = 4
    classifier_encoder, test_df = get_encoded_df(
        log=test_log,
        encoder=classifier_encoder,
        CONF=CONF_TEST,
        train_df=classifier_train_df
    )
    recommendations = recommender_model.model.recommend(test_df, top_n=3)
    result = evaluate_recommender(test_df['label'], recommendations)

    logger.info('Done, cheers!')

    return { 'result': result}


if __name__ == '__main__':
    dic = run_recommender_pipeline()
    print(str(dic))


