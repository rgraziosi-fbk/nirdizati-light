'''
classe process che contiene risorse ed esegue task
'''
import simpy
from rims.resource_priority import ResourcePriority
import math
from rims.call_LSTM import Predictor


class SimulationProcess(object):

    def __init__(self, env: simpy.Environment, params):
        self.env = env
        self.params = params
        self.date_start = params.START_SIMULATION
        self.single_resources = self.define_single_resources()
        self.resource_events = self.define_resource_events(env)
        self.resource_trace = simpy.Resource(env, math.inf)
        self.buffer_traces = []
        self.predictor = Predictor((params.MODEL_PATH_PROCESSING, params.MODEL_PATH_WAITING), self.params)
        self.predictor.predict()

    #def define_roles(self):
    #    set_resource = list(self.params.ROLE_CAPACITY.keys())
    #    dict_res = dict()
    #    for res in set_resource:
    #        res_simpy = Resource(self.env, res, self.params.ROLE_CAPACITY[res][0], self.params.ROLE_CAPACITY[res][1], self.date_start)
    #        dict_res[res] = res_simpy
    #    return dict_res

    def get_occupations_resource(self, resource):
        occup = []
        if self.params.FEATURE_ROLE == 'all_role':
            for key in self.roles:
                if key != 'SYSTEM':
                    occup.append(self.roles[key].get_resource().count / self.roles[key].capacity)
        else:
            occup.append(self.roles[resource].get_resource().count / self.roles[resource].capacity)
        return occup

    def define_single_resources(self):
        dict_res = dict()
        for res in self.params.SINGLE_RESOURCES:
            res_simpy = ResourcePriority(self.env, res, 1, {'days': [0, 1, 2, 3, 4, 5, 6], 'hour_min': 0, 'hour_max': 23},
                                 self.date_start)
            dict_res[res] = res_simpy

        return dict_res

    def get_single_resource(self, resource_label):
        return self.single_resources[resource_label]

    def get_occupations_all_role(self, role):
        """
        Method to retrieve the occupancy in percentage of all roles, as an intercase feature.
        """
        occup = []
        if self.params.FEATURE_ROLE == 'all_role':
            for key in self.params.ROLE_CAPACITY:
                if key != 'SYSTEM':
                    occup.append(self.get_occupations_single_role_LSTM(key))
        else:
            occup.append(self.get_occupations_single_role_LSTM(role))
        return occup

    def get_occupations_single_role_LSTM(self, role):
        """
        Method to retrieve the specified role occupancy in percentage, as an intercase feature:
        $\\frac{resources \: occupated \: in \:role}{total\:resources\:in\:role}$.
        """
        occup = 0
        for res in self.single_resources:
            if self.params.RESOURCE_TO_ROLE_LSTM[res] == role:
                occup += self.single_resources[res].get_resource().count
        occup = occup / self.params.ROLE_CAPACITY[role][0]
        return round(occup, 2)


    #def get_role(self, resource_label):
    #    return self.roles[resource_label]

    def get_resource_event(self, task):
        return self.resource_events[task]

    def get_resource_trace(self):
        return self.resource_trace

    def define_resource_events(self, env):
        resources = dict()
        for key in self.params.INDEX_AC:
            resources[key] = simpy.Resource(env, math.inf)
        return resources

    def get_predict_processing(self, cid, pr_wip, transition, ac_wip, rp_oc, time, queue):
        return self.predictor.processing_time(cid, pr_wip, transition, ac_wip, rp_oc, time, queue)

    def get_predict_waiting(self, cid, pr_wip, transition, rp_oc, time, queue):
        if queue < 0:
            return self.predictor.predict_waiting(cid, pr_wip, transition, rp_oc, time, queue)
        else:
            return self.predictor.predict_waiting_queue(cid, pr_wip, transition, rp_oc, time, queue)
