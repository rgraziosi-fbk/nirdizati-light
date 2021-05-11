from enum import Enum

from pm4py.objects.log.log import EventLog


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


def get_prefix_length(trace_len: int, prefix_length: float) -> int:
	if prefix_length >= 1:
		return int(prefix_length)
	else:
		return int(prefix_length*trace_len)


def get_max_prefix_length(log: EventLog, prefix_length: float) -> int:
	if prefix_length > 1:
		return int(prefix_length)
	prefix_lengths = [get_prefix_length(len(trace), prefix_length) for trace in log]
	prefix_lengths.append(int(prefix_length))
	max_prefix_length = max(prefix_lengths)
	return max_prefix_length
