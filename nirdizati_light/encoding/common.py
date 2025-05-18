import logging
from enum import Enum

from pandas import DataFrame,Series,to_datetime,Timestamp
from pm4py.objects.log.obj import EventLog
from datetime import datetime
from nirdizati_light.encoding.data_encoder import Encoder
from nirdizati_light.encoding.feature_encoder.complex_features import complex_features
from nirdizati_light.encoding.feature_encoder.frequency_features import frequency_features
from nirdizati_light.encoding.feature_encoder.loreley_complex_features import loreley_complex_features
from nirdizati_light.encoding.feature_encoder.loreley_features import loreley_features
from nirdizati_light.encoding.feature_encoder.simple_features import simple_features
from nirdizati_light.encoding.feature_encoder.binary_features import binary_features
from nirdizati_light.encoding.feature_encoder.simple_trace_features import simple_trace_features
# from src.encoding.feature_encoder.declare_features.declare_features import declare_features
from nirdizati_light.encoding.time_encoding import time_encoding

logger = logging.getLogger(__name__)


class EncodingType(Enum):
    SIMPLE = 'simple'
    FREQUENCY = 'frequency'
    COMPLEX = 'complex'
    DECLARE = 'declare'
    LORELEY = 'loreley'
    LORELEY_COMPLEX = 'loreley_complex'
    SIMPLE_TRACE = 'simple_trace'
    BINARY = 'binary'

class EncodingTypeAttribute(Enum):
    LABEL = 'label'
    ONEHOT = 'onehot'


TRACE_TO_DF = {
    EncodingType.SIMPLE.value : simple_features,
    EncodingType.FREQUENCY.value : frequency_features,
    EncodingType.COMPLEX.value : complex_features,
    # EncodingType.DECLARE.value : declare_features,
    EncodingType.LORELEY.value: loreley_features,
    EncodingType.LORELEY_COMPLEX.value: loreley_complex_features,
    EncodingType.SIMPLE_TRACE.value: simple_trace_features,
    EncodingType.BINARY.value: binary_features,

}


def get_encoded_df(
        log: EventLog, CONF: dict=None, encoder: Encoder=None, 
        train_cols: DataFrame=None, train_df=None) -> (Encoder, DataFrame):
    """
    Encode log with the configuration provided in the CONF dictionary.

    :param EventLog log: EventLog object of the log
    :param dict CONF: dictionary for configuring the encoding
    :param nirdizati_light.encoding.data_encoder.Encoder: if an encoder is provided, that encoder will be used instead of creating a new one

    :return: A tuple containing the encoder and the encoded log as a Pandas dataframe
    """

    logger.debug('SELECT FEATURES')
    df = TRACE_TO_DF[CONF['feature_selection']](
        log,
        prefix_length=CONF['prefix_length'],
        padding=CONF['padding'],
        prefix_length_strategy=CONF['prefix_length_strategy'],
        labeling_type=CONF['labeling_type'],
        generation_type=CONF['task_generation_type'],
        feature_list=train_cols,
        target_event=CONF['target_event'],
    )
    logger.debug('EXPLODE DATES')

    def compute_durations(df):
        for prefix in range(CONF['prefix_length']):
            prefix += 1  # Start at 1

            duration_col = f'duration_{prefix}'
            waiting_col = f'waiting_{prefix}'
            arrival_col = f'arrival_{prefix}'
            timestamp_col = f'time:timestamp_{prefix}'
            start_timestamp_col = f'start:timestamp_{prefix}'
            previous_timestamp_col = f'time:timestamp_{prefix - 1}' if prefix - 1 > 0 else None
            next_start_timestamp_col = f'start:timestamp_{prefix + 1}' if prefix + 1 <= CONF['prefix_length'] else None

            # Convert timestamps to datetime, handling errors safely
            df[timestamp_col] = to_datetime(df[timestamp_col], errors='coerce', utc=True)
            df[start_timestamp_col] = to_datetime(df[start_timestamp_col], errors='coerce', utc=True).fillna(
                "1970-01-01 00:00+00")
            df[timestamp_col] = to_datetime(df[timestamp_col], errors='coerce', utc=True).fillna(
                "1970-01-01 00:00+00")


            # Compute duration (convert timedelta to seconds)
            try:
                df[duration_col] = (df[timestamp_col] - df[start_timestamp_col]).dt.total_seconds()
                df[duration_col] = df[duration_col].fillna(0).apply(lambda x: max(x, 0))  # Handle NaN and negatives
            except Exception as e:
                logger.error(f"Error computing duration for prefix {prefix}: {e}")
                df[duration_col] = 0

            if prefix > 1 and previous_timestamp_col in df.columns:
                df[previous_timestamp_col] = to_datetime(df[previous_timestamp_col], errors='coerce')
                try:
                    df[waiting_col] = (df[timestamp_col] - df[previous_timestamp_col]).dt.total_seconds()
                except Exception as e:
                    logger.error(f"Error computing waiting time for prefix {prefix}: {e}")
                    df[waiting_col] = 0
                df[waiting_col] = df[waiting_col].fillna(0).apply(lambda x: max(x, 0))  # Handle NaN and negatives
            else:
                df[waiting_col] = 0

            if next_start_timestamp_col and next_start_timestamp_col in df.columns:
                #df[next_start_timestamp_col] = to_datetime(df[next_start_timestamp_col], errors='coerce').fillna(
                #    "1970-01-01 00:00+00")
                df[next_start_timestamp_col] = (
                    to_datetime(df[next_start_timestamp_col], errors='coerce', utc=True)
                    .fillna(to_datetime("1970-01-01 00:00+00"))
                )
                try:
                    df[arrival_col] = (df[next_start_timestamp_col] - df[timestamp_col]).dt.total_seconds()
                except Exception as e:
                    logger.error(f"Error computing arrival for prefix {prefix}: {e}")
                    df[arrival_col] = 0
                df[arrival_col] = df[arrival_col].fillna(0).apply(lambda x: max(x, 0))  # Handle NaN and negatives
            else:
                df[arrival_col] = 0  # If there's no next start timestamp
            # Insert columns at the correct positions
            df.insert(df.columns.get_loc(timestamp_col) + 1, waiting_col, df.pop(waiting_col))
            idx = df.columns.get_loc(timestamp_col) + 1  # Insert after timestamp column
            df.insert(idx, duration_col, df.pop(duration_col))
            df.insert(idx + 1, arrival_col, df.pop(arrival_col))
            if 'start:timestamp_1' in df.columns:
                df.insert(df.columns.get_loc('prefix_1'), 'start_trace', df.pop('start:timestamp_1'))
        return df
    if CONF['feature_selection'] == 'complex' and CONF['explanator'] == 'dice_augmentation':
        df = compute_durations(df)


    df = time_encoding(df, CONF['time_encoding'])



    if CONF['feature_selection'] == 'complex' and CONF['explanator'] == 'dice_augmentation':
        df = df[df.columns[~Series(df.columns).str.contains(
            'cases|time|queue|open|group|event|lifecycle|day|hour|week|month')]]

    logger.debug('ALIGN DATAFRAMES')
    if train_df is not None:
        _, df = train_df.align(df, join='left', axis=1)

    if not encoder:
        logger.debug('INITIALISE ENCODER')
        encoder = Encoder(df=df, attribute_encoding=CONF['attribute_encoding'],feature_selection=CONF['feature_selection'],
                          prefix_length=CONF['prefix_length'])
    logger.debug('ENCODE')
    encoder.encode(df=df)

    return encoder, df
