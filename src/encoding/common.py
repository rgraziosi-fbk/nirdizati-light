import logging
from enum import Enum

from pandas import DataFrame
from pm4py.objects.log.log import EventLog

from src.encoding.data_encoder import Encoder
# from src.encoding.feature_encoder.frequency_features import frequency_features
from src.encoding.feature_encoder.simple_features import simple_features
from src.encoding.feature_encoder.complex_features import complex_features
# from src.encoding.feature_encoder.declare_features.declare_features import declare_features

logger = logging.getLogger(__name__)


class EncodingType(Enum):
    SIMPLE = 'simple'
    FREQUENCY = 'frequency'
    COMPLEX = 'complex'
    DECLARE = 'declare'

class EncodingTypeAttribute(Enum):
    LABEL = 'label'
    ONEHOT = 'onehot'


class EncodingTypeAttribute(Enum):
    LABEL = 'label'
    ONEHOT = 'onehot'


TRACE_TO_DF = {
    EncodingType.SIMPLE.value : simple_features,
    # EncodingType.FREQUENCY.value : frequency_features,
    EncodingType.COMPLEX.value : complex_features,
    # EncodingType.DECLARE.value : declare_features
}


def get_encoded_df(train_log: EventLog, validate_log: EventLog, test_log: EventLog, retrain_test_log: EventLog, CONF: dict=None) -> (DataFrame, DataFrame, DataFrame):
    logger.debug('SELECT FEATURES')
    train_df = TRACE_TO_DF[CONF['feature_selection']](
        train_log,
        prefix_length=CONF['prefix_length'],
        padding=CONF['padding'],
        labeling_type=CONF['labeling_type'],
        feature_list=None
    )
    validate_df = TRACE_TO_DF[CONF['feature_selection']](
        validate_log,
        prefix_length=CONF['prefix_length'],
        padding=CONF['padding'],
        labeling_type=CONF['labeling_type'],
        feature_list=train_df.columns)
    test_df = TRACE_TO_DF[CONF['feature_selection']](
        test_log,
        prefix_length=CONF['prefix_length'],
        padding=CONF['padding'],
        labeling_type=CONF['labeling_type'],
        feature_list=train_df.columns
    )
    retrain_test_df = TRACE_TO_DF[CONF['feature_selection']](
        retrain_test_log,
        prefix_length=CONF['prefix_length'],
        padding=CONF['padding'],
        labeling_type=CONF['labeling_type'],
        feature_list=train_df.columns
    )

    logger.debug('INITIALISE ENCODER')
    encoder = Encoder(df=train_df, attribute_encoding=CONF['attribute_encoding'])

    logger.debug('ENCODE')
    encoder.encode(df=train_df)
    encoder.encode(df=validate_df)
    encoder.encode(df=test_df)
    encoder.encode(df=retrain_test_df)

    return encoder, train_df, validate_df, test_df, retrain_test_df
