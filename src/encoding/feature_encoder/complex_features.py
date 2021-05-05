from functools import reduce

from pandas import DataFrame
from pm4py.objects.log.log import Trace, EventLog

from src.labeling.common import add_label_column

ATTRIBUTE_CLASSIFIER = None

PREFIX_ = 'prefix_'


def complex_features(log: EventLog, prefix_length, padding, labeling_type, feature_list: list = None) -> DataFrame:
    columns, additional_columns = _columns_complex(log, prefix_length, feature_list)
    encoded_data = []
    for trace in log:
        if len(trace) <= prefix_length - 1 and not padding:
            # trace too short and no zero padding
            continue
        encoded_data.append(_trace_to_row(trace, prefix_length, additional_columns, padding, columns, labeling_type))

    return DataFrame(columns=columns, data=encoded_data)


def _get_global_trace_attributes(log: EventLog):
    # retrieves all traces in the log and returns their intersection
    attributes = list(reduce(set.intersection, [set(trace._get_attributes().keys()) for trace in log]))
    trace_attributes = [attr for attr in attributes if attr not in ["concept:name", "time:timestamp", "label"]]
    return sorted(trace_attributes)


def _get_global_event_attributes(log):
    """Get log event attributes that are not name or time
    """
    # retrieves all events in the log and returns their intersection
    attributes = list(reduce(set.intersection, [set(event._dict.keys()) for trace in log for event in trace]))
    event_attributes = [attr for attr in attributes if attr not in ["concept:name", "time:timestamp"]]
    return sorted(event_attributes)


def _compute_additional_columns(log) -> dict:
    return {'trace_attributes': _get_global_trace_attributes(log),
            'event_attributes': _get_global_event_attributes(log)}


def _columns_complex(log, prefix_length: int, feature_list: list = None) -> tuple:
    additional_columns = _compute_additional_columns(log)
    columns = ['trace_id']
    columns += additional_columns['trace_attributes']
    for i in range(1, prefix_length + 1):
        columns.append(PREFIX_ + str(i))
        for additional_column in additional_columns['event_attributes']:
            columns.append(additional_column + "_" + str(i))
    columns += ['label']
    if feature_list is not None:
        assert(list(feature_list) == columns)
    return columns, additional_columns


def _data_complex(trace: Trace, prefix_length: int, additional_columns: dict) -> list:
    """Creates list in form [1, value1, value2, 2, ...]

    Appends values in additional_columns
    """
    data = [trace.attributes.get(att, 0) for att in additional_columns['trace_attributes']]
    for idx, event in enumerate(trace):
        if idx == prefix_length:
            break
        event_name = event["concept:name"]
        data.append(event_name)

        for att in additional_columns['event_attributes']:
            data.append(event.get(att, '0'))

    return data


def _trace_to_row(trace: Trace, prefix_length: int, additional_columns, padding, columns: list, labeling_type) -> list:
    trace_row = [trace.attributes["concept:name"]]
    trace_row += _data_complex(trace, prefix_length, additional_columns)
    if padding:
        trace_row += [0 for _ in range(len(trace_row), len(columns) - 1)]
    trace_row += [ add_label_column(trace, labeling_type, prefix_length) ]
    return trace_row
