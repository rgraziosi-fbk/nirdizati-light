import logging

from pandas import DataFrame
from pm4py.objects.log.log import EventLog

from src.encoding.data_encoder import Encoder
from src.encoding.feature_encoder.frequency_features import frequency_features
from src.encoding.feature_encoder.simple_features import simple_features
from src.encoding.feature_encoder.complex_features import complex_features
from src.encoding.constants import EncodingType
# from src.encoding.feature_encoder.declare_features.declare_features import declare_features
from src.encoding.time_encoding import time_encoding, TimeEncodingType

logger = logging.getLogger(__name__)


TRACE_TO_DF = {
    EncodingType.SIMPLE.value : simple_features,
    EncodingType.FREQUENCY.value : frequency_features,
    EncodingType.COMPLEX.value : complex_features,
    # EncodingType.DECLARE.value : declare_features
}


def get_encoded_df(log: EventLog, CONF: dict=None, encoder: Encoder=None, train_cols: DataFrame=None, train_df=None) -> (Encoder, DataFrame):
    logger.debug('SELECT FEATURES')
    df = TRACE_TO_DF[CONF['feature_selection']](
        log,
        prefix_length=CONF['prefix_length'],
        padding=CONF['padding'],
        prefix_length_strategy=CONF['prefix_length_strategy'],
        labeling_type=CONF['labeling_type'],
        generation_type=CONF['task_generation_type'],
        feature_list=train_cols,
        target_event=CONF['target_event']
    )

    logger.debug('EXPLODE DATES')
    if CONF['time_encoding'] != TimeEncodingType.NONE.value:
        df = time_encoding(df, CONF['time_encoding'])

    logger.debug('ALIGN DATAFRAMES')
    if train_df is not None:
        _, df = train_df.align(df, join='left', axis=1)

    if not encoder:
        logger.debug('INITIALISE ENCODER')
        encoder = Encoder(df=df, attribute_encoding=CONF['attribute_encoding'])

    logger.debug('ENCODE')
    encoder.encode(df=df)

    return encoder, df
