from pandas import DataFrame
from pm4py.objects.log.log import EventLog, Trace

from src.labeling.common import add_label_column

ATTRIBUTE_CLASSIFIER = None
PREFIX_ = 'prefix_'


def simple_features(log: EventLog, prefix_length, padding, labeling_type, feature_list: list = None) -> DataFrame:
    columns = _compute_columns(prefix_length)
    encoded_data = []
    for trace in log:
        if len(trace) <= prefix_length - 1 and not padding:
            # trace too short and no zero padding
            continue
        encoded_data.append(_trace_to_row(trace, prefix_length, padding, labeling_type))

    return DataFrame(columns=columns, data=encoded_data)


def _trace_to_row(trace: Trace, prefix_length: int, padding: bool = True, labeling_type: str = None) -> list:
    """Row in data frame"""
    trace_row = [trace.attributes['concept:name']]
    trace_row += _trace_prefixes(trace, prefix_length)
    if padding:
        trace_row += [0 for _ in range(len(trace_row), prefix_length + 1)]
    trace_row += [ add_label_column(trace, labeling_type, prefix_length) ]
    return trace_row


def _trace_prefixes(trace: Trace, prefix_length: int) -> list:
    """List of indexes of the position they are in event_names

    """
    prefixes = []
    for idx, event in enumerate(trace):
        if idx == prefix_length:
            break
        event_name = event['concept:name']
        prefixes.append(event_name)
    return prefixes


def _compute_columns(prefix_length: int) -> list:
    """trace_id, prefixes, any other columns, label

    """
    return ["trace_id"] + [PREFIX_ + str(i + 1) for i in range(0, prefix_length)] + ['label']

