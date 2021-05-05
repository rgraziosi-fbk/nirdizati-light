from collections import Counter
from datetime import timedelta

from pandas import DataFrame
from pm4py.objects.log.log import EventLog, Trace, Event

from src.labeling.common import add_label_column

PREFIX_ = 'prefix_'


def frequency_features(log: EventLog, prefix_length, padding, labeling_type, columns: list = None) -> DataFrame:
    if columns is None:
        columns = _compute_columns(log, prefix_length)
    encoded_data = []
    for trace in log:
        if len(trace) <= prefix_length - 1 and not padding:
            # trace too short and no zero padding
            continue
        encoded_data.append(_trace_to_row(trace, prefix_length, columns, padding, labeling_type))

    return DataFrame(columns=columns, data=encoded_data)


def _compute_columns(log: EventLog, prefix_length: int, padding: bool) -> list:
    """trace_id, prefixes, any other columns, label

    """
    return ["trace_id"] + \
           sorted(list({
               event['concept:name']
               for trace in log
               for event in trace[:prefix_length]
           })) + \
           ['0'] if padding else [] + \
           ['label']


def _trace_to_row(trace: Trace, prefix_length: int, columns: list, padding: bool = True, labeling_type: str = None) -> list:
    """Row in data frame"""
    trace_row = [ trace.attributes['concept:name'] ]

    if len(trace) <= prefix_length - 1 and not padding:
        trace += [
            Event({
                'concept:name': '0',
                'time:timestamp': trace[len(trace)] + timedelta(hours=i)
            })
            for i in range(len(trace), prefix_length + 1)
        ]

    occurences = Counter([
        event['concept:name']
        for event in trace[:prefix_length]
    ])
    trace_row += [ occurences[col] for col in columns ]
    trace_row += [ add_label_column(trace, labeling_type, prefix_length) ]
    return trace_row

