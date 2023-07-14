from enum import Enum


class LabelTypes(Enum):
    NEXT_ACTIVITY = 'next_activity'
    ATTRIBUTE_STRING = 'label_attribute_string'


def add_label_column(trace, labeling_type, prefix_length: int):
    """TODO COMMENT ME
    """
    if labeling_type == LabelTypes.NEXT_ACTIVITY.value:
        return next_event_name(trace, prefix_length)
    elif labeling_type == LabelTypes.ATTRIBUTE_STRING.value:
        return trace.attributes['label']
    else:
        raise Exception('Label not set please select one of LabelTypes(Enum) values!')


def next_event_name(trace: list, prefix_length: int):
    """Return the event event_name at prefix length or 0 if out of range.

    """
    if prefix_length < len(trace):
        next_event = trace[prefix_length]
        name = next_event['concept:name']
        return name
    else:
        return 0


