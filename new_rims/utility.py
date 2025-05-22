"""
* Prefix: Class to handle the prefix shared by events within traces.
* Buffer: Class to handle the features of a single event required for predictive models.
"""

import os


def define_folder_output(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class ParallelObject(object):

    def __init__(self):
        self._am_parallel = []

    def _set_last_events(self, am):
        for token in am:
            self._am_parallel.append(token)

    def _get_last_events(self):
        tokens = set(self._am_parallel)
        self._am_parallel = []
        return tokens

    def _update_last_events(self):
        self._am_parallel = []


class Prefix(object):

    def __init__(self):
        self._list_activities = list()

    def add_activity(self, activity):
        self._list_activities.append(activity)

    def get_prefix(self):
        return self._list_activities


class Buffer(object):

    def __init__(self, writer, TRACE_ATTRIBUTES=None, EVENT_ATTRIBUTES=None, values=None):
        self.buffer = {"id_case": -1, "activity": None, "role": None, "enabled_time": None, "start_time": None,
                             "end_time": None, "resource": None, "prefix": Prefix} | {a: None for a in EVENT_ATTRIBUTES} | {a: None for a in
                                                                                       TRACE_ATTRIBUTES}
        if values:
            self._decopy_value(values)
        self.writer = writer
        self._TRACE_ATTRIBUTES = TRACE_ATTRIBUTES
        self._EVENT_ATTRIBUTES = EVENT_ATTRIBUTES

    def _decopy_value(self, values):
        for key in values:
            self.buffer[key] = values[key]

    def _get_dictionary(self):
        return self.buffer

    def set_feature(self, feature, value):
        if isinstance(self.buffer[feature], list):
            self.buffer[feature] = value
        else:
            self.buffer[feature] = value

    def get_feature(self, feature):
        return self.buffer[feature]

    def print_values(self):
        print(*self.buffer.values())
        self.writer.writerow(self.buffer.values())

    def get_buffer_keys(self):
        return self.buffer.keys()

    def reset(self):
        self.buffer["activity"] = None
        self.buffer["role"] = None
        self.buffer["enabled_time"] = None
        self.buffer["start_time"] = None
        self.buffer["end_time"] = -1
        for a in self._EVENT_ATTRIBUTES:
            self.buffer[a] = None
        for a in self._TRACE_ATTRIBUTES:
            self.buffer[a] = None