import logging
from enum import Enum

from pandas import DataFrame
from pm4py.objects.log.obj import EventLog

from nirdizati_light.encoding.constants import PrefixLengthStrategy, TaskGenerationType
from nirdizati_light.encoding.data_encoder import Encoder
from nirdizati_light.encoding.feature_encoder.complex_features import complex_features
from nirdizati_light.encoding.feature_encoder.frequency_features import frequency_features
from nirdizati_light.encoding.feature_encoder.loreley_complex_features import loreley_complex_features
from nirdizati_light.encoding.feature_encoder.loreley_features import loreley_features
from nirdizati_light.encoding.feature_encoder.simple_features import simple_features
from nirdizati_light.encoding.feature_encoder.binary_features import binary_features
from nirdizati_light.encoding.feature_encoder.simple_trace_features import simple_trace_features
from nirdizati_light.encoding.time_encoding import TimeEncodingType, time_encoding
from nirdizati_light.labeling.common import LabelTypes

logger = logging.getLogger(__name__)


class EncodingType(Enum):
    SIMPLE = 'simple'
    FREQUENCY = 'frequency'
    COMPLEX = 'complex'
    LORELEY = 'loreley'
    LORELEY_COMPLEX = 'loreley_complex'
    SIMPLE_TRACE = 'simple_trace'
    BINARY = 'binary'

class EncodingTypeAttribute(Enum):
    LABEL = 'label'
    ONEHOT = 'onehot'

ENCODE_LOG = {
    EncodingType.SIMPLE.value : simple_features,
    EncodingType.FREQUENCY.value : frequency_features,
    EncodingType.COMPLEX.value : complex_features,
    EncodingType.LORELEY.value: loreley_features,
    EncodingType.LORELEY_COMPLEX.value: loreley_complex_features,
    EncodingType.SIMPLE_TRACE.value: simple_trace_features,
    EncodingType.BINARY.value: binary_features,

}

def get_encoded_df(
    log: EventLog,
    encoder: Encoder = None,
    feature_encoding_type=EncodingType.SIMPLE.value,
    prefix_length=10,
    prefix_length_strategy=PrefixLengthStrategy.FIXED.value,
    time_encoding_type=TimeEncodingType.NONE.value,
    attribute_encoding=EncodingTypeAttribute.LABEL.value,
    padding=True,
    labeling_type=LabelTypes.ATTRIBUTE_STRING.value,
    task_generation_type=TaskGenerationType.ONLY_THIS.value,
    target_event=None,
    train_cols: DataFrame=None,
    train_df=None
) -> tuple[Encoder, DataFrame]:
    """
    Encode log with the configuration provided in the CONF dictionary.

    :param EventLog log: EventLog object of the log
    :param dict CONF: dictionary for configuring the encoding
    :param nirdizati_light.encoding.data_encoder.Encoder: if an encoder is provided, that encoder will be used instead of creating a new one

    :return: A tuple containing the encoder and the encoded log as a Pandas dataframe
    """

    logger.debug(f'Features encoding ({feature_encoding_type})')
    df = ENCODE_LOG[feature_encoding_type](
        log,
        prefix_length=prefix_length,
        padding=padding,
        prefix_length_strategy=prefix_length_strategy,
        labeling_type=labeling_type,
        generation_type=task_generation_type,
        feature_list=train_cols,
        target_event=target_event,
    )

    logger.debug(f'Time encoding ({time_encoding_type})')
    df = time_encoding(df, time_encoding_type)

    logger.debug('Dataframe alignment')
    if train_df is not None:
        _, df = train_df.align(df, join='left', axis=1)

    if not encoder:
        logger.debug('Encoder initialization')
        encoder = Encoder(df=df, attribute_encoding=attribute_encoding, prefix_length=prefix_length)

    logger.debug('Encoding')
    encoder.encode(df=df)

    return encoder, df
