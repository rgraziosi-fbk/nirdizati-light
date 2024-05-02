from datetime import datetime
import csv
import simpy
from rims.checking_process import SimulationProcess
from rims.token_LSTM import Token
from rims.MAINparameters import Parameters
import sys, getopt
import warnings
from os.path import exists
import os
import pandas as pd
import pm4py
from pm4py.objects.log.util import sorting
from pm4py.objects.log.importer.xes import importer as xes_importer

SEPSIS_ATTRIB_TRACE = ['Age', 'Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG', 'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
                 'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture', 'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg', 'Hypotensie',
                 'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie', 'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea', 'SIRSCritTemperature', 'SIRSCriteria2OrMore']

SEPSIS_ATTRIB_EVENT = ['CRP', 'LacticAcid', 'Leucocytes', 'event_nr', 'hour', 'month', 'timesincecasestart', 'timesincelastevent', 'timesincemidnight', 'weekday']

def read_log_csv(self, path):
    dataframe = pd.read_csv(path, sep=',')
    dataframe = pm4py.format_dataframe(dataframe, case_id='caseid', activity_key='task',
                                       timestamp_key='end_timestamp')
    event_log = pm4py.convert_to_event_log(dataframe)
    event_log = sorting.sort_timestamp(event_log, timestamp_key='start_timestamp')
    return event_log

def read_training(train, attrib_event, attrib_trace):
    # [event, event_processingTime, resource, wait, amount]
    #train = pd.read_csv(path, sep=',')
    arrivals_train = []
    resource = 'org:group_'
    columns = list(train.columns)
    count_prefix = 1
    traces = dict()
    for index, row in train.iterrows():
        prefix = 'prefix_' + str(count_prefix)
        start = pd.to_datetime(row['start:timestamp_1'], unit='s')
        key = str(row['trace_id'])
        arrivals_train.append([key, start])
        buffer = []
        attributes_trace = {}
        attributes_event = {}
        while prefix in columns and row[prefix] != '0' and row[prefix] != 0:
            start = row['start:timestamp_'+str(count_prefix)]
            end = row['time:timestamp_'+str(count_prefix)]
            processing = end - start
            ###### aggiungere qua la copia degli attributi di evento!!!!
            if count_prefix <= 1:
                wait = 0
            else:
                start = row['start:timestamp_' + str(count_prefix)]
                end = row['time:timestamp_'+str(count_prefix-1)]
                wait = start - end
            for k in attrib_event:
                attributes_event[k] = row[k + '_' + str(count_prefix)]
            for k in attrib_trace:
                attributes_trace[k] = row[k]
            buffer.append([row[prefix], processing, row[resource + str(count_prefix)], wait, attributes_event, attributes_trace, row['label']])
            count_prefix += 1
            prefix = 'prefix_' + str(count_prefix)

        traces[key] = buffer
        count_prefix = 1
    return traces, arrivals_train


def read_contrafactual(contrafactual, attrib_event, attrib_trace):
    ## list of events = [[activity, resource], ....]
    #contrafactual = pd.read_csv(path, sep=',')
    arrivals_CF = []
    resource = 'org:group_'
    columns = list(contrafactual.columns)
    count_prefix = 1
    contrafactual_traces = dict()
    for index, row in contrafactual.iterrows():
        key = str(row['trace_id']) + "_CF"
        contrafactual_traces[key] = []
        prefix = 'prefix_' + str(count_prefix)
        start = pd.to_datetime(row['start:timestamp_1'], unit='s')
        arrivals_CF.append([key, start])
        attributes_trace = {}
        attributes_event = {}
        for k in attrib_trace:
            attributes_trace[k] = row[k]
        for k in attrib_event:
            attributes_event[k] = row[k + '_' + str(count_prefix)]
        while prefix in columns and row[prefix] != '0' and row[prefix] != 0:
            contrafactual_traces[key].append(
                [row[prefix], row[resource + str(count_prefix)], attributes_event, attributes_trace, row['label']])
            count_prefix += 1
            prefix = 'prefix_' + str(count_prefix)
        count_prefix = 1
    return contrafactual_traces, arrivals_CF

def setup(env: simpy.Environment, NAME_EXPERIMENT, params, i, type, log, arrivals, contrafactual, key):
    simulation_process = SimulationProcess(env=env, params=params)
    path_result = os.getcwd() + '/' + NAME_EXPERIMENT + '/results/simulated_log_' + NAME_EXPERIMENT + '_' + '.csv'
    f = open(path_result, 'w')
    writer = csv.writer(f)
    writer.writerow(['caseid', 'task', 'arrive:timestamp', 'start:timestamp', 'time:timestamp', 'role', 'open_cases', 'st_tsk_wip', 'queue'] + SEPSIS_ATTRIB_EVENT + ['attrib_trace', 'label'])
    prev = params.START_SIMULATION
    for i in range(0, len(arrivals)):
        next = arrivals[i][1]
        interval = (next - prev).total_seconds()
        prev = next
        yield env.timeout(interval)
        if str(arrivals[i][0]) in key:
            id_arrival = str(arrivals[i][0])
            env.process(Token(id_arrival, params, simulation_process, params, [], contrafactual[arrivals[i][0]].copy()).simulation(env, writer, type))
        else:
            env.process(Token(arrivals[i][0], params, simulation_process, params, log[arrivals[i][0]].copy(), False).simulation(env, writer, type))


def run(NAME_EXPERIMENT, type, log, arrivals, contrafactual, key):
    path_model = os.getcwd() + '/' + NAME_EXPERIMENT + '/' + NAME_EXPERIMENT
    if exists(path_model + '_diapr_meta.json'):
        FEATURE_ROLE = 'all_role'
    elif exists(path_model + '_dispr_meta.json'):
        FEATURE_ROLE = 'no_all_role'
    else:
        raise ValueError('LSTM models do not exist in the right folder')
    N_SIMULATION = 1
    for i in range(0, N_SIMULATION):
        params = Parameters(NAME_EXPERIMENT, FEATURE_ROLE, i, type)
        env = simpy.Environment()
        env.process(setup(env, NAME_EXPERIMENT, params, i, type, log, arrivals, contrafactual, key))
        env.run(until=params.SIM_TIME)

def run_simulation(train_df, df_cf, NAME_EXPERIMENT = 'sepsis_cases_1_start', type ='rims', N_SIMULATION = 1):
    print(NAME_EXPERIMENT, N_SIMULATION, type)
    log, arrivals = read_training(train_df, SEPSIS_ATTRIB_EVENT, SEPSIS_ATTRIB_TRACE)
    contrafactual_traces, arrivals_CF = read_contrafactual(df_cf, SEPSIS_ATTRIB_EVENT, SEPSIS_ATTRIB_TRACE)
    arrivals = sorted(arrivals + arrivals_CF, key=lambda x: x[1])
    run(NAME_EXPERIMENT, type, log, arrivals, contrafactual_traces, list(contrafactual_traces.keys()))