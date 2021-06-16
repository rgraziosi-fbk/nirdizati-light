from enum import Enum

from pm4py.objects.log.log import EventLog, Trace


class EncodingType(Enum):
	SIMPLE = 'simple'
	FREQUENCY = 'frequency'
	COMPLEX = 'complex'
	DECLARE = 'declare'


class EncodingTypeAttribute(Enum):
	LABEL = 'label'
	ONEHOT = 'onehot'


class TaskGenerationType(Enum):
	ONLY_THIS = 'only_this'
	ALL_IN_ONE = 'all_in_one'


class PrefixLengthStrategy(Enum):
	FIXED = 'fixed'
	PERCENTAGE = 'percentage'


def get_prefix_length(trace: Trace, prefix_length: float, prefix_length_strategy, target_event=None) -> int:
	if prefix_length_strategy == PrefixLengthStrategy.FIXED.value:
		return int(prefix_length)
	else:
		return int(prefix_length*trace_len)


def get_max_prefix_length(log: EventLog, prefix_length: float, prefix_length_strategy, target_event) -> int:
	prefix_lengths = [get_prefix_length(trace, prefix_length, prefix_length_strategy, target_event) for trace in log]
	return max(prefix_lengths)
