from datetime import datetime, timedelta
import simpy
import pm4py
import random
from new_rims.process import SimulationProcess
from pm4py.objects.petri_net import semantics
from new_rims.parameters import Parameters
from new_rims.utility import Prefix
from simpy.events import AnyOf, AllOf, Event
import copy
import numpy as np
from scipy.stats import expon
import math
import csv
from new_rims.utility import Buffer, ParallelObject
from scipy.stats import lognorm, truncnorm
import new_rims.custom_function as custom

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

class Token(object):

    def __init__(self, id: int,
                 params: Parameters, process: SimulationProcess, prefix: Prefix, type: str, writer: csv.writer,
                 parallel_object: ParallelObject, time: datetime, sequence, NAME_EXPERIMENT,  TRACE_ATTRIBUTES, EVENT_ATTRIBUTES, CF, values=None):
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
        self._TRACE_ATTRIBUTES = TRACE_ATTRIBUTES
        self._EVENT_ATTRIBUTES = EVENT_ATTRIBUTES
        self._buffer = Buffer(writer,  TRACE_ATTRIBUTES, EVENT_ATTRIBUTES)
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
                                          self._writer, self._parallel_object, self._start_time, [next],
                                          self.NAME_EXPERIMENT, self._TRACE_ATTRIBUTES, self._EVENT_ATTRIBUTES, self.CF).simulation(env))
                next_events = [next, token]
                for t in next[-1]:
                    token = env.process(Token(self._id, self._params, self._process, self._prefix, "parallel",
                                              self._writer, self._parallel_object, self._start_time, [t],
                                          self.NAME_EXPERIMENT, self._TRACE_ATTRIBUTES, self._EVENT_ATTRIBUTES, self.CF).simulation(env))
                    next_events.append(token)
                del next[-1]
            else:
                next_events = next
            del self.sequence[0]
        else:
            next_events = None
        return next_events


    def predict_processing_time(self, res, act, time, event):
        ## input features: ["caseid", "org:resource", "concept:name", "weekday", "hour"] + TRACE_ATTRIBUTES + EVENT_ATTRIBUTES + PREFIX_COLUMNS
        input = [self._params.RESOURCE_2_NUMBER[res], self._params.ACT_2_NUMBER[act],
                 time.weekday(), time.hour]
        ### for sepsis
        for t in self._params.TRACES_ATTRIBUTES: #int(bool('True'))
            if t == 'DIAGNOSE':
                value = self._params.DIAGNOSE_2_NUMBER[event[-1][t]]
                input += [value]
            else:
                input += [int(bool(event[-1][t]))]
        for e in self._params.EVENT_ATTRIBUTES:
            input += [0 if event[-2][e] == 'other' else int(event[-2][e])]

        actual_prefix = self._prefix.get_prefix()
        for p in range(self._params.PREFIX_LEN):
            if p < len(actual_prefix):
                input += [self._params.ACT_2_NUMBER[actual_prefix[p]]]
            else:
                input += [self._params.ACT_2_NUMBER['PAD']]

        input_array = np.array(input).reshape(1, -1)
        mean_pred = self._process.mean_processing.predict(input_array)
        input_std = np.hstack((input_array, mean_pred.reshape(1, -1)))
        std_pred = self._process.std_processing.predict(input_std)
        mu_rescaled = self._process.scalar_processing.inverse_transform(mean_pred.reshape(-1, 1))[0][0]
        sigma_rescaled = self._process.scalar_processing.inverse_transform(std_pred.reshape(-1, 1))[0][0]

        proc_time_pred = round(np.random.normal(mu_rescaled, sigma_rescaled))
        return max(0, proc_time_pred)


    def predict_waiting_time(self, res, act, time, event):
        ## input features: ["caseid", "org:resource", "concept:name", "weekday", "hour"] + TRACE_ATTRIBUTES + EVENT_ATTRIBUTES + PREFIX_COLUMNS
        input = [self._params.RESOURCE_2_NUMBER[res], self._params.ACT_2_NUMBER[act],
                 time.weekday(), time.hour]
        ### for sepsis
        for t in self._params.TRACES_ATTRIBUTES: #int(bool('True'))
            if t == 'DIAGNOSE':
                value = self._params.DIAGNOSE_2_NUMBER[event[-1][t]]
                input += [value]
            else:
                input += [int(bool(event[-1][t]))]
        for e in self._params.EVENT_ATTRIBUTES:
            input += [0 if event[-2][e] == 'other' else int(event[-2][e])]

        actual_prefix = self._prefix.get_prefix()
        for p in range(self._params.PREFIX_LEN):
            if p < len(actual_prefix):
                input += [self._params.ACT_2_NUMBER[actual_prefix[p]]]
            else:
                input += [self._params.ACT_2_NUMBER['PAD']]

        input_array = np.array(input).reshape(1, -1)
        mean_pred = self._process.mean_waiting.predict(input_array)
        input_std = np.hstack((input_array, mean_pred.reshape(1, -1)))
        std_pred = self._process.std_waiting.predict(input_std)
        mu_rescaled = self._process.scalar_waiting.inverse_transform(mean_pred.reshape(-1, 1))[0][0]
        sigma_rescaled = self._process.scalar_waiting.inverse_transform(std_pred.reshape(-1, 1))[0][0]

        wait_time_pred = round(np.random.normal(mu_rescaled, sigma_rescaled))
        return max(0, wait_time_pred)

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
            if event is not None:
                self._buffer.reset()
                self._buffer.set_feature("id_case", self._id)
                self._buffer.set_feature("activity", event[1])
                self._buffer.set_feature("prefix", self._prefix.get_prefix())

                #### attribute events
                for e in self._params.EVENT_ATTRIBUTES:
                    self._buffer.set_feature(e, event[-2][e])
                ### attribute traces
                for t in self._params.TRACES_ATTRIBUTES:
                    self._buffer.set_feature(t, event[-1][t])

                # event: sequence/parallel, task, processing_time, resource, wait, event_attrib, event_event
                name_res = self.define_role(self._params.ACT_TO_ROLE[event[1]]) if self.CF else self._params.RES_TO_ROLE[event[3]]
                #name_res = self._params.ACT_TO_ROLE[event[1]] if self.CF else self._params.RES_TO_ROLE[event[3]]
                resource = self._process._get_resource(name_res)
                self._buffer.set_feature("role", resource._get_name())

                ### register event in process ###
                resource_task = self._process._get_resource_event(event[1])

                queue = 0 if len(resource._queue) == 0 else len(resource._queue[-1])
                self._buffer.set_feature("enabled_time", self._start_time + timedelta(seconds=env.now))

                request_resource = resource.request()
                yield request_resource
                single_resource = self._process._set_single_resource(resource._get_name())
                self._buffer.set_feature("resource", single_resource)

                resource_task_request = resource_task.request()
                yield resource_task_request

                ## calendars
                stop = resource.to_time_schedule(self._start_time + timedelta(seconds=env.now))
                yield env.timeout(stop)
                self._buffer.set_feature("start_time", self._start_time + timedelta(seconds=env.now))
                if self.CF:
                    duration = self.define_processing_time(event[1])
                else:
                    duration = event[2]

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

    def define_role(self, list_roles):
        role = list_roles
        if isinstance(list_roles, list):
            free_role = []
            for res in list_roles:
                if self._process._get_resource(res)._resource_simpy.count < self._process._get_resource(res)._capacity:
                    free_role.append(res)
            if not free_role:
                role = random.choice(list_roles)
            else:
                role = random.choice(free_role)
        return role

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

    def define_processing_time(self, act):
        ### call the RF, put all the encoding
        if self._params.PROCESSING_TIME[act]["name"] == 'lognorm':
            mean = self._params.PROCESSING_TIME[act]["parameters"]["mean"]
            variance = self._params.PROCESSING_TIME[act]["parameters"]["std"]
            min_val = self._params.PROCESSING_TIME[act]["parameters"]["min"]
            max_val = self._params.PROCESSING_TIME[act]["parameters"]["max"]
            sigma = np.sqrt(np.log(1 + (variance / mean ** 2)))
            mu = np.log(mean) - 0.5 * sigma ** 2
            def truncated_lognorm(mu, sigma, min_val, max_val, size=1000):
                a, b = (np.log(min_val + 1e-9) - mu) / sigma, (np.log(max_val) - mu) / sigma
                samples = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size)
                return np.exp(samples)
            duration = truncated_lognorm(mu, sigma, min_val, max_val, size=1)[0]
        elif self._params.PROCESSING_TIME[act]["name"] == "exponential":
            scale = self._params.PROCESSING_TIME[act]["parameters"]["scale"]
            min_val = self._params.PROCESSING_TIME[act]["parameters"]["min"]
            max_val = self._params.PROCESSING_TIME[act]["parameters"]["max"]
            def truncated_exponential_inverse(scale, min_val, max_val, size=1000):
                cdf_min = expon.cdf(min_val, scale=scale)
                cdf_max = expon.cdf(max_val, scale=scale)
                u = np.random.uniform(cdf_min, cdf_max, size=size)
                return expon.ppf(u, scale=scale)
            duration = truncated_exponential_inverse(scale, min_val, max_val, size=1)[0]
        elif self._params.PROCESSING_TIME[act]["name"] == 'norm':
            mean = self._params.PROCESSING_TIME[act]["parameters"]["mean"]
            std = self._params.PROCESSING_TIME[act]["parameters"]["std"]
            min_val = self._params.PROCESSING_TIME[act]["parameters"]["min"]
            max_val = self._params.PROCESSING_TIME[act]["parameters"]["max"]
            a = (min_val - mean) / std
            b = (max_val - mean) / std
            duration = truncnorm.rvs(a, b, loc=mean, scale=std)
        else:
            distribution = self._params.PROCESSING_TIME[act]['name']
            parameters = self._params.PROCESSING_TIME[act]['parameters']
            duration = getattr(np.random, distribution)(**parameters, size=1)[0]
            if duration < 0:
                print("WARNING: Negative processing time", duration)
                duration = 0

        return duration

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
