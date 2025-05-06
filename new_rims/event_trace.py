from datetime import datetime, timedelta
import simpy
import pm4py
import random
from process import SimulationProcess
from pm4py.objects.petri_net import semantics
from parameters import Parameters
from utility import Prefix
from simpy.events import AnyOf, AllOf, Event
import copy
import csv
from utility import Buffer, ParallelObject
import custom_function as custom

class Token(object):

    def __init__(self, id: int,
                 params: Parameters, process: SimulationProcess, prefix: Prefix, type: str, writer: csv.writer,
                 parallel_object: ParallelObject, time: datetime, sequence, NAME_EXPERIMENT, buffer_definition, CF, values=None):
        self._id = id
        self._process = process
        self._start_time = params.START_SIMULATION
        self._params = params
        self._prefix = prefix
        self._type = type
        if type == 'sequential':
            self.see_activity = False
        else:
            self.see_activity = True
        self._writer = writer
        self._parallel_object = parallel_object
        self._buffer_definition = buffer_definition
        self._buffer = Buffer(writer, buffer_definition)
        ### added
        self.pos = 0
        self.sequence = sequence
        self.CF = CF
        self.NAME_EXPERIMENT = NAME_EXPERIMENT

    def next_event(self, env: simpy.Environment):  ### add the consideration of parallel
        if self.sequence:
            next = self.sequence[0]
            if next[0] == True: ### parallel
                next[0] = False
                token = env.process(Token(self._id, self._params, self._process, self._prefix, "parallel",
                                          self._writer, self._parallel_object, self._buffer._get_dictionary(), [next],
                                          self.CF, self.NAME_EXPERIMENT, self._buffer_definition).simulation(env))
                next_events = [next, token]
                for t in next[-1]:
                    token = env.process(Token(self._id, self._params, self._process, self._prefix, "parallel",
                                              self._writer, self._parallel_object, self._buffer._get_dictionary(), [t],
                                              self.CF, self.NAME_EXPERIMENT, self._buffer_definition).simulation(env))
                    next_events.append(token)
                del next[-1]
                #del self.sequence[0]
                #after_parallel = self.sequence[0]
                #next_events.insert(0, after_parallel)
            else:
                next_events = next
            del self.sequence[0]
        else:
            next_events = None
        return next_events


    def simulation(self, env: simpy.Environment):
        """
            The main function to handle the simulation of a single trace
        """
        ### register trace in process ###
        event = self.next_event(env)
        request_resource = None
        resource_trace = self._process._get_resource_trace()
        resource_trace_request = resource_trace.request() if self._type == 'sequential' else None

        while event is not None:
            if not self.see_activity and self._type == 'sequential':
                yield resource_trace_request
            if type(event[0]) == list: ### check parallel
                yield AllOf(env, event[1:])
                event = self.next_event(env)
                #event = event[0]
            if event is not None:
                self._buffer.reset()
                self._buffer.set_feature("id_case", self._id)
                self._buffer.set_feature("activity", event[1])
                self._buffer.set_feature("prefix", self._prefix.get_prefix(self._start_time + timedelta(seconds=env.now)))

                #### attribute events
                for e in self._params.EVENT_ATTRIBUTES:
                    self._buffer.set_feature(e, event[-2][e])
                ### attribute traces
                for t in self._params.TRACES_ATTRIBUTES:
                    self._buffer.set_feature(t, event[-1][t])

                # event: sequence/parallel, task, processing_time, resource, wait, event_attrib, event_event
                name_res = event[2] if self.CF else event[3]
                resource = self._process._get_resource(name_res)
                self._buffer.set_feature("role", resource._get_name())

                ### register event in process ###
                resource_task = self._process._get_resource_event(event[1])

                queue = 0 if len(resource._queue) == 0 else len(resource._queue[-1])
                self._buffer.set_feature("enabled_time", self._start_time + timedelta(seconds=env.now))

                waiting = 0 if self.CF else event[4] #### to adjust with the prediction
                if self.see_activity:
                    yield env.timeout(waiting)

                request_resource = resource.request()
                yield request_resource
                single_resource = self._process._set_single_resource(resource._get_name())
                self._buffer.set_feature("resource", single_resource)

                resource_task_request = resource_task.request()
                yield resource_task_request

                #stop = resource.to_time_schedule(self._start_time + timedelta(seconds=env.now))
                #yield env.timeout(stop)
                self._buffer.set_feature("start_time", self._start_time + timedelta(seconds=env.now))
                duration = 0 if self.CF else event[2] #### to adjust with the prediction

                yield env.timeout(duration)

                self._buffer.set_feature("end_time", self._start_time + timedelta(seconds=env.now))
                self._buffer.print_values()
                self._prefix.add_activity(event[1])
                resource.release(request_resource)
                self._process._release_single_resource(resource._get_name(), single_resource)
                resource_task.release(resource_task_request)
                event = self.next_event(env)

            if self._type == 'sequential':
                resource_trace.release(resource_trace_request)

    def _get_resource_role(self, activity):
        elements = self._params.ROLE_ACTIVITY[activity.label]
        resource_object = []
        for e in elements:
            resource_object.append(self._process._get_resource(e))
        return resource_object

    def define_xor_next_activity(self, all_enabled_trans):
        """ Three different methods to decide which path following from XOR gateway:
        * Random choice: each path has equal probability to be chosen (AUTO)
        ```json
        "probability": {
            "A_ACCEPTED": "AUTO",
            "skip_2": "AUTO",
            "A_FINALIZED": "AUTO",
        }
        ```
        * Defined probability: in the file json it is possible to define for each path a specific probability (PROBABILITY as value)
        ```json
        "probability": {
            "A_PREACCEPTED": 0.20,
            "skip_1": 0.80
        }
        ```
        * Custom method: it is possible to define a dedicate method that given the possible paths it returns the one to
        follow, using whatever techniques the user prefers. (CUSTOM)
        ```json
        "probability": {
            "A_CANCELLED": "CUSTOM",
            "A_DECLINED": "CUSTOM",
            "tauSplit_5": "CUSTOM"
        }
        ```
        """
        prob = ['AUTO'] if not self._params.PROBABILITY else self._retrieve_check_paths(all_enabled_trans)
        self._check_type_paths(prob)
        if prob[0] == 'AUTO':
                next = random.choices(list(range(0, len(all_enabled_trans), 1)))[0]
        elif prob[0] == 'CUSTOM':
            next = self.call_custom_xor_function(all_enabled_trans)
        elif type(prob[0] == float()):
            if self._check_probability(prob):
                value = [*range(0, len(prob), 1)]
                next = int(random.choices(value, prob)[0])
            else:
                next = random.choices(list(range(0, len(all_enabled_trans), 1)))[0]

        return all_enabled_trans[next]

    def define_processing_time(self, activity):
        ### call the RF, put all the encoding
        return 0

    def define_waiting_time(self, next_act):
        ### call the RF, put all the encoding
        return 0

    ### modify to consider the parallel and delete the petrinet logic
    def next_transition(self, env):
        """
        Method to define the next activity in the petrinet.
        """
        all_enabled_trans = semantics.enabled_transitions(self._net, self._am)
        all_enabled_trans = list(all_enabled_trans)
        all_enabled_trans.sort(key=lambda x: x.name)
        if len(all_enabled_trans) == 0:
            return None
        elif len(all_enabled_trans) == 1:
            return all_enabled_trans[0]
        else:
            if len(self._am) == 1:
                return self.define_xor_next_activity(all_enabled_trans)
            else:
                events = []
                for token in self._am:
                    name = token.name
                    new_am = copy.copy(self._am)
                    tokens_to_delete = self._delete_tokens(name)
                    for p in tokens_to_delete:
                        del new_am[p]
                    path = env.process(Token(self._id, self._net, new_am, self._params, self._process, self._prefix, "parallel", self._writer, self._parallel_object, self._buffer._get_dictionary()).simulation(env))
                    events.append(path)
                return events
