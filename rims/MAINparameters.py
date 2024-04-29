'''
Principal parameters to run the process

DA AGGIUNGERE: configurazione risorse, tempi per ogni attivita'
'''
import json
import os
from datetime import datetime

import pandas as pd


class Parameters(object):

    def __init__(self, name_exp, feature_role, iterations, type):
        self.NAME_EXP = name_exp
        self.FEATURE_ROLE = feature_role
        self.PATH_PETRINET = os.getcwd() + '/rims/' + self.NAME_EXP + '/' + self.NAME_EXP + '.pnml'
        if self.FEATURE_ROLE == 'all_role':
            self.prefix = ('_diapr', '_dpiapr', '_dwiapr')
        else:
            self.prefix = ('_dispr', '_dpispr', '_dwispr')

        self.MODEL_PATH_PROCESSING = os.getcwd() + '/rims/' + self.NAME_EXP + '/' + self.NAME_EXP + self.prefix[1] +'.h5'
        self.MODEL_PATH_WAITING = os.getcwd() + '/rims/' + self.NAME_EXP + '/' + self.NAME_EXP + self.prefix[2] + '.h5'

        self.SIM_TIME = 1460*36000000000000000  # 10 day
        self.METADATA = os.getcwd() + '/rims/' + self.NAME_EXP + '/' + self.NAME_EXP + self.prefix[0] + '_meta.json'
        self.SCALER = os.getcwd() + '/rims/' + self.NAME_EXP + '/' + self.NAME_EXP + self.prefix[0] + '_scaler.pkl'
        self.INTER_SCALER = os.getcwd() + '/rims/' + self.NAME_EXP + '/' + self.NAME_EXP + self.prefix[0] + '_inter_scaler.pkl'
        self.END_INTER_SCALER = os.getcwd() + '/rims/' + self.NAME_EXP + '/' + self.NAME_EXP + self.prefix[0] + '_end_inter_scaler.pkl'
        self.read_metadata_file()

    def read_metadata_file(self):
        print('METADATA', self.METADATA)
        if os.path.exists(self.METADATA):
            with open(self.METADATA) as file:
                data = json.load(file)
                self.INDEX_AC = data['ac_index']
                self.AC_WIP_INITIAL = data['inter_mean_states']['tasks']
                self.PR_WIP_INITIAL = round(data['inter_mean_states']['wip'])
                self.START_SIMULATION = datetime.strptime(data["start_simulation"], '%Y-%m-%d %H:%M:%S')
                roles_table = data['roles_table']
                self.ROLE_ACTIVITY = dict()
                for elem in roles_table:
                    self.ROLE_ACTIVITY[elem['task']] = elem['role']

                self.INDEX_ROLE = {'SYSTEM': 0}
                self.ROLE_CAPACITY = {'SYSTEM': [1000, {'days': [0, 1, 2, 3, 4, 5, 6], 'hour_min': 0, 'hour_max': 23}]}
                self.RESOURCE_ROLE = {}
                for role in data['roles']:
                    for res in data['roles'][role]:
                        self.RESOURCE_ROLE[res] = role
                roles = data['roles']
                for idx, key in enumerate(roles):
                    self.INDEX_ROLE[key] = idx
                    self.ROLE_CAPACITY[key] = [len(roles[key]), {'days': [0, 1, 2, 3, 4, 5, 6], 'hour_min': 0, 'hour_max': 23}]
                self.SINGLE_RESOURCES = data['resources']
                self.RESOURCE_TO_ROLE_LSTM = {}
                for idx, key in enumerate(roles):
                    for resource in roles[key]:
                        self.RESOURCE_TO_ROLE_LSTM[resource] = key

