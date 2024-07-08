from pandas import DataFrame
from pm4py.objects.log.obj import EventLog, Trace

from nirdizati_light.encoding.constants import TaskGenerationType, get_prefix_length, get_max_prefix_length, PrefixLengthStrategy
from nirdizati_light.labeling.common import add_label_column

ATTRIBUTE_CLASSIFIER = None
PREFIX_ = 'prefix_'


def simple_features(log: EventLog, prefix_length: int, padding: bool, prefix_length_strategy: str, labeling_type: str, generation_type: str, feature_list=None, target_event: str = None) -> DataFrame:
    """Generates a DataFrame with simple features from an event log."""
    max_prefix_length = get_max_prefix_length(log, prefix_length, prefix_length_strategy, target_event)
    columns = _generate_columns(max_prefix_length)
    encoded_data = _generate_encoded_data(log, prefix_length, padding, prefix_length_strategy, labeling_type, generation_type, target_event, columns)

    return DataFrame(columns=columns, data=encoded_data)


def _generate_encoded_data(log, prefix_length, padding, prefix_length_strategy, labeling_type, generation_type, target_event, columns):
    """Generates encoded data for the DataFrame."""
    encoded_data = []
    for trace in log:
        trace_prefix_length = get_prefix_length(trace, prefix_length, prefix_length_strategy, target_event)
        if _should_skip_trace(trace, prefix_length, padding):
            continue
        encoded_data += _encode_trace(trace, trace_prefix_length, generation_type, columns, prefix_length_strategy, padding, labeling_type)
    return encoded_data


def _should_skip_trace(trace, prefix_length, padding):
    """Determines if a trace should be skipped based on its length and padding."""
    return len(trace) <= prefix_length - 1 and not padding


def _encode_trace(trace, trace_prefix_length, generation_type, columns, prefix_length_strategy, padding, labeling_type):
    """Encodes a single trace into a row or rows for the DataFrame."""
    encoded_rows = []
    if generation_type == TaskGenerationType.ALL_IN_ONE.value:
        for event_index in range(1, min(trace_prefix_length + 1, len(trace) + 1)):
            encoded_rows.append(_trace_to_row(trace, event_index, len(columns), prefix_length_strategy, padding, labeling_type))
    else:
        encoded_rows.append(_trace_to_row(trace, trace_prefix_length, len(columns), prefix_length_strategy, padding, labeling_type))
    return encoded_rows


def _trace_to_row(trace: Trace, prefix_length: int, columns_number: int, prefix_length_strategy: str, padding: bool = True, labeling_type: str = None) -> list:
    """Converts a trace to a row for the DataFrame."""
    trace_row = [trace.attributes['concept:name']] + _trace_prefixes(trace, prefix_length)
    trace_row += _pad_trace_row(trace_row, columns_number, padding, prefix_length_strategy)
    trace_row += [add_label_column(trace, labeling_type, prefix_length)]
    return trace_row


def _trace_prefixes(trace: Trace, prefix_length: int) -> list:
    """Extracts prefixes from a trace."""
    return [event['concept:name'] for idx, event in enumerate(trace) if idx < prefix_length]


def _pad_trace_row(trace_row, columns_number, padding, prefix_length_strategy):
    """Pads a trace row to match the expected number of columns."""
    if padding or prefix_length_strategy == PrefixLengthStrategy.PERCENTAGE.value:
        return [0 for _ in range(len(trace_row), columns_number - 1)]
    return []


def _generate_columns(prefix_length: int) -> list:
    """Generates column names for the DataFrame."""
    return ["trace_id"] + [PREFIX_ + str(i + 1) for i in range(prefix_length)] + ['label']