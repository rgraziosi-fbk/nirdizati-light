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

SEPSIS_ATTRIB = ['Age', 'Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG', 'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
                 'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture', 'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg', 'Hypotensie',
                 'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie', 'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea', 'SIRSCritTemperature', 'SIRSCriteria2OrMore']


def read_log_csv(self, path):
    dataframe = pd.read_csv(path, sep=',')
    dataframe = pm4py.format_dataframe(dataframe, case_id='caseid', activity_key='task',
                                       timestamp_key='end_timestamp')
    event_log = pm4py.convert_to_event_log(dataframe)
    event_log = sorting.sort_timestamp(event_log, timestamp_key='start_timestamp')
    return event_log

def define_log_to_check(path):
    # [event, event_processingTime, resource, wait]
    #event_log = pm4py.read_xes('test_log_one_lifecycle.xes')
    event_log = xes_importer.apply(path)
    event_log = sorting.sort_timestamp(event_log, timestamp_key='start:timestamp')
    log = dict()
    arrivals = []
    for idx, trace in enumerate(event_log):
        trace_sim = []
        arrival = datetime.strptime(str(trace[0]['start:timestamp'])[:19], '%Y-%m-%d %H:%M:%S')
        arrivals.append([trace.attributes['concept:name'], arrival])
        for idx_e, event in enumerate(trace):
            if event['concept:name'] != 'Start' and event['concept:name'] != 'End':
                i = 1
                if (idx_e == 0):
                    wait = 0
                else:
                    start = datetime.strptime(str(trace[idx_e]['start:timestamp'])[:19], '%Y-%m-%d %H:%M:%S')
                    end_bef = datetime.strptime(str(trace[idx_e - i]['time:timestamp'])[:19], '%Y-%m-%d %H:%M:%S')
                    wait = (start - end_bef).total_seconds()
                while wait < 0 and idx_e - i > 1:
                    i += 1
                    if (idx_e - i == 0):
                        wait = 0
                    else:
                        start = datetime.strptime(str(trace[idx_e]['start:timestamp'])[:19], '%Y-%m-%d %H:%M:%S')
                        end_bef = datetime.strptime(str(trace[idx_e - i]['time:timestamp'])[:19],'%Y-%m-%d %H:%M:%S')
                        wait = (start - end_bef).total_seconds()
                if wait < 0:
                    wait = 0
                start = datetime.strptime(str(trace[idx_e]['start:timestamp'])[:19], '%Y-%m-%d %H:%M:%S')
                end = datetime.strptime(str(event['time:timestamp'])[:19], '%Y-%m-%d %H:%M:%S')
                duration = (end - start).total_seconds()
                trace_sim.append([event['concept:name'], duration, event['org:resource'], wait, trace.attributes['AMOUNT_REQ']])
        log[trace.attributes['concept:name']] = trace_sim
    return log, arrivals

def read_training(path, attrib):
    # [event, event_processingTime, resource, wait, amount]
    train = pd.read_csv(path, sep=',')
    arrivals_train = []
    resource = 'org:resource_'
    columns = list(train.columns)
    count_prefix = 1
    traces = dict()
    for index, row in train.iterrows():
        prefix = 'prefix_' + str(count_prefix)
        start = pd.to_datetime(row['start:timestamp_1'], unit='s')
        key = str(row['trace_id'])
        arrivals_train.append([key, start])
        buffer = []
        attributes = {}
        while prefix in columns and row[prefix] != '0' and row[prefix] != 0:
            start = row['start:timestamp_'+str(count_prefix)]
            end = row['time:timestamp_'+str(count_prefix)]
            processing = end - start

            if count_prefix <= 1:
                wait = 0
            else:
                start = row['start:timestamp_' + str(count_prefix)]
                end = row['time:timestamp_'+str(count_prefix-1)]
                wait = start - end
            for k in attrib:
                attributes[k] = row[k]
            buffer.append([row[prefix], processing, row[resource + str(count_prefix)], wait, attributes])
            count_prefix += 1
            prefix = 'prefix_' + str(count_prefix)

        traces[key] = buffer
        count_prefix = 1
    return traces, arrivals_train


def read_contrafactual(path, attrib):
    ## list of events = [[activity, resource], ....]
    contrafactual = pd.read_csv(path, sep=',')
    arrivals_CF = []
    resource = 'org:resource_'
    columns = list(contrafactual.columns)
    count_prefix = 1
    contrafactual_traces = dict()
    for index, row in contrafactual.iterrows():
        key = str(row['trace_id']) + "_CF"
        contrafactual_traces[key] = []
        prefix = 'prefix_' + str(count_prefix)
        start = pd.to_datetime(row['start:timestamp_1'], unit='s')
        arrivals_CF.append([key, start])
        attributes = {}
        for k in attrib:
            attributes[k] = row[k]
        while prefix in columns and row[prefix] != '0' and row[prefix] != 0:
            contrafactual_traces[key].append(
                [row[prefix], row[resource + str(count_prefix)], attributes])
            count_prefix += 1
            prefix = 'prefix_' + str(count_prefix)
        count_prefix = 1
    return contrafactual_traces, arrivals_CF

def main(argv):
    opts, args = getopt.getopt(argv, "h:t:l:n:")
    NAME_EXPERIMENT = 'confidential_1000'
    for opt, arg in opts:
        if opt == '-h':
            print('main.py -t <[rims, rims_plus]> -l <log_name> -n <total number of simulation [1, 25]>')
            sys.exit()
        elif opt == "-t":
            type = arg
        elif opt == "-l":
            NAME_EXPERIMENT = arg
        elif opt == "-n":
            N_SIMULATION = int(arg)
            if N_SIMULATION > 25:
                N_SIMULATION = 25
    print(NAME_EXPERIMENT, N_SIMULATION, type)
    log, arrivals = read_training('rims/sepsis_cases_1/sepsis_cases_1_start_cf_complete_rem_time.csv', SEPSIS_ATTRIB)
    contrafactual_traces, arrivals_CF = read_contrafactual('rims/sepsis_cases_1/sepsis_cases_1_start_train_df_complete_rem_time.csv', SEPSIS_ATTRIB)
    arrivals = sorted(arrivals + arrivals_CF, key=lambda x: x[1])
    run_simulation(NAME_EXPERIMENT, type, log, arrivals, contrafactual_traces, list(contrafactual_traces.keys()))


def setup(env: simpy.Environment, NAME_EXPERIMENT, params, i, type, log, arrivals, contrafactual, key):
    simulation_process = SimulationProcess(env=env, params=params)
    path_result = os.getcwd() + '/rims/' + NAME_EXPERIMENT + '/results/simulated_log_' + NAME_EXPERIMENT + '_' + '.csv'
    f = open(path_result, 'w')
    writer = csv.writer(f)
    writer.writerow(['caseid', 'task', 'arrive:timestamp', 'start:timestamp', 'time:timestamp', 'role', 'st_wip', 'st_tsk_wip', 'queue', 'amount'])
    prev = params.START_SIMULATION
    for i in range(0, 5):
        next = arrivals[i][1]
        interval = (next - prev).total_seconds()
        prev = next
        yield env.timeout(interval)
        if str(arrivals[i][0]) in key:
            id_arrival = str(arrivals[i][0])
            env.process(Token(id_arrival, params, simulation_process, params, [], contrafactual[arrivals[i][0]].copy()).simulation(env, writer, type))
        else:
            env.process(Token(arrivals[i][0], params, simulation_process, params, log[arrivals[i][0]].copy(), False).simulation(env, writer, type))


def run_simulation(NAME_EXPERIMENT, type, log, arrivals, contrafactual, key):
    path_model = os.getcwd() + '/rims/' + NAME_EXPERIMENT + '/' + NAME_EXPERIMENT
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

def run():
    NAME_EXPERIMENT = 'sepsis_cases_1_start'
    type = 'rims'
    N_SIMULATION = 1
    print(NAME_EXPERIMENT, N_SIMULATION, type)
    log, arrivals = read_training(os.getcwd() + '/rims/sepsis_cases_1_start/sepsis_cases_1_start_cf_complete_rem_time.csv', SEPSIS_ATTRIB)
    contrafactual_traces, arrivals_CF = read_contrafactual(os.getcwd() + '/rims/sepsis_cases_1_start/sepsis_cases_1_start_train_df_complete_rem_time.csv', SEPSIS_ATTRIB)
    arrivals = sorted(arrivals + arrivals_CF, key=lambda x: x[1])
    run_simulation(NAME_EXPERIMENT, type, log, arrivals, contrafactual_traces, list(contrafactual_traces.keys()))

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    #main(sys.argv[1:])
    run()
