import logging
from enum import Enum
from typing import Optional

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
    """
    Available trace encoding types
    """
    SIMPLE = 'simple'
    FREQUENCY = 'frequency'
    COMPLEX = 'complex'
    LORELEY = 'loreley'
    LORELEY_COMPLEX = 'loreley_complex'
    SIMPLE_TRACE = 'simple_trace'
    BINARY = 'binary'

class EncodingTypeAttribute(Enum):
    """
    Available trace attributes encoding types
    """
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
    encoder: Optional[Encoder] = None,
    feature_encoding_type: EncodingType = EncodingType.SIMPLE.value,
    prefix_length: int = 10,
    prefix_length_strategy: PrefixLengthStrategy = PrefixLengthStrategy.FIXED.value,
    time_encoding_type: TimeEncodingType = TimeEncodingType.NONE.value,
    attribute_encoding: EncodingTypeAttribute = EncodingTypeAttribute.LABEL.value,
    padding: bool = True,
    labeling_type: LabelTypes = LabelTypes.ATTRIBUTE_STRING.value,
    task_generation_type: TaskGenerationType = TaskGenerationType.ONLY_THIS.value,
    target_event: Optional[str] = None,
    train_cols: Optional[DataFrame] = None,
    train_df: Optional[DataFrame] = None,
) -> tuple[Encoder, DataFrame]:
    """
    Encodes an event log into a DataFrame using specified encoding configurations.

    This method allows for the customization of the encoding process through various parameters, including the type of feature encoding, prefix length, time encoding, and more.

    The method returns a tuple containing the encoder used and the resulting DataFrame.

    Args:
        log (EventLog): The event log to be encoded.
        encoder (Optional[Encoder]): The encoder to be used. If None, a default encoder based on the feature encoding type will be used.
        feature_encoding_type (EncodingType): The type of feature encoding to use. Defaults to EncodingType.SIMPLE.
        prefix_length (int): The length of the prefix to consider for each case. Defaults to 10.
        prefix_length_strategy (PrefixLengthStrategy): The strategy to use for prefix length (e.g., fixed, percentage). Defaults to PrefixLengthStrategy.FIXED.
        time_encoding_type (TimeEncodingType): The type of time encoding to use. Defaults to TimeEncodingType.NONE.
        attribute_encoding (EncodingTypeAttribute): The type of attribute encoding to use. Defaults to EncodingTypeAttribute.LABEL.
        padding (bool): Whether to pad sequences to a fixed length. Defaults to True.
        labeling_type (LabelTypes): The type of labeling to use for the encoded log. Defaults to LabelTypes.ATTRIBUTE_STRING.
        task_generation_type (TaskGenerationType): The type of task generation to use. Defaults to TaskGenerationType.ONLY_THIS.
        target_event (Optional[str]): The target event to consider for encoding. Defaults to None.
        train_cols (Optional[DataFrame]): The DataFrame containing the training columns. Defaults to None.
        train_df (Optional[DataFrame]): The training DataFrame. Defaults to None.

    Returns:
        Tuple[Encoder, DataFrame]: A tuple containing the encoder and the encoded DataFrame.
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
