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
from operator import itemgetter
# Check if eager execution is enabled

ATTRIBUTES = {
        'sepsis_cases_1_start': {'TRACE': ['Age', 'Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG', 'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
                 'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture', 'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg', 'Hypotensie',
                 'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie', 'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea', 'SIRSCritTemperature', 'SIRSCriteria2OrMore'], 'EVENT': ['CRP', 'LacticAcid', 'Leucocytes', 'event_nr', 'hour', 'month', 'timesincecasestart', 'timesincelastevent', 'timesincemidnight', 'weekday']},
        'BPI_Challenge_2012_W_Two_TS':{'TRACE': ['AMOUNT_REQ'], 'EVENT': []},
        'bpic2015_2_start': {'TRACE': ['Aanleg (Uitvoeren werk of werkzaamheid)', 'Bouw',
                                             'Brandveilig gebruik (melding)', 'Brandveilig gebruik (vergunning)',
                                             'Gebiedsbescherming', 'Handelen in strijd met regels RO',
                                             'Inrit/Uitweg', 'Kap', 'Milieu (melding)',
                                             'Milieu (neutraal wijziging)',
                                             'Milieu (omgevingsvergunning beperkte milieutoets)',
                                             'Milieu (vergunning)', 'Monument', 'Reclame', 'Responsible_actor',
                                             'SUMleges', 'Sloop'], 'EVENT': ['event_nr', 'hour','lifecycle:transition', 'month',
                                                                             'question', 'timesincecasestart',
                                                                             'timesincelastevent', 'timesincemidnight',
                                                                                 'weekday']},
        'sepsis_cases_2_start': {
          'TRACE': ['Age', 'Diagnose', 'DiagnosticArtAstrup','DiagnosticBlood','DiagnosticECG',
             'DiagnosticIC','DiagnosticLacticAcid','DiagnosticLiquor','DiagnosticOther',
             'DiagnosticSputum','DiagnosticUrinaryCulture','DiagnosticUrinarySediment',
             'DiagnosticXthorax','DisfuncOrg','Hypotensie','Hypoxie',
             'InfectionSuspected','Infusion','Oligurie','SIRSCritHeartRate',
             'SIRSCritLeucos','SIRSCritTachypnea','SIRSCritTemperature','SIRSCriteria2OrMore'],
           'EVENT': ['CRP', 'LacticAcid', 'Leucocytes', 'event_nr', 'hour', 'month',
                                'timesincecasestart',
                                'timesincelastevent', 'timesincemidnight', 'weekday']},
        'sepsis_cases_3_start': {
                  'TRACE': ['Age', 'Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG',
                            'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
                            'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture',
                            'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg', 'Hypotensie',
                            'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie', 'SIRSCritHeartRate',
                            'SIRSCritLeucos', 'SIRSCritTachypnea', 'SIRSCritTemperature', 'SIRSCriteria2OrMore'],
                  'EVENT': ['CRP', 'LacticAcid', 'Leucocytes', 'event_nr', 'hour', 'month', 'timesincecasestart',
                            'timesincelastevent', 'timesincemidnight', 'weekday']},
        'bpic2015_4_start': {'TRACE': ['Aanleg (Uitvoeren werk of werkzaamheid)', 'Bouw','Brandveilig gebruik (vergunning)',
                                                             'Gebiedsbescherming', 'Handelen in strijd met regels RO',
                                                             'Inrit/Uitweg', 'Kap',
                                                             'Milieu (neutraal wijziging)',
                                                             'Milieu (omgevingsvergunning beperkte milieutoets)',
                                                             'Milieu (vergunning)', 'Monument', 'Reclame', 'Responsible_actor',
                                                             'SUMleges', 'Sloop'],
                             'EVENT': ['event_nr', 'hour','lifecycle:transition', 'month',
                                       'question', 'timesincecasestart','timesincelastevent', 'timesincemidnight',
                                                                                                 'weekday']},
        'bpic2012_2_start_old': {'TRACE': ['AMOUNT_REQ'],'EVENT': ["hour", "weekday", "month", "timesincemidnight",
                                                                    "timesincelastevent",
                                                                    "timesincecasestart", "event_nr"]},
        'bpic2012_2_start': {'TRACE': ['AMOUNT_REQ'],'EVENT': ["hour", "weekday", "month", "timesincemidnight",
                                                                            "timesincelastevent",
                                                                            "timesincecasestart", "event_nr"]},
        'Productions': {'TRACE': ["Part_Desc_", "Report_Type", "Rework",
                                              "Work_Order_Qty"],
                        'EVENT': ["Qty_Completed", "Qty_for_MRB", "activity_duration", "event_nr",
                        "hour", "lifecycle:transition", "month", "timesincecasestart", "timesincelastevent",
                                  "timesincemidnight", "weekday"]},
        'PurchasingExample': {'TRACE': ['lifecycle:transition'],
                                'EVENT': ["event_nr",
                                "hour", "month", "timesincecasestart", "timesincelastevent",
                                          "timesincemidnight", "weekday"]},
        'ConsultaDataMining201618': {'TRACE': ['lifecycle:transition'],
                                          'EVENT': ["event_nr",
                                                    "hour", "month", "timesincecasestart",
                                                    "timesincelastevent",
                                                    "timesincemidnight", "weekday"]},
        'cvs_pharmacy': {'TRACE': ['lifecycle:transition'],
                                          'EVENT': ["event_nr", "resourceCost",
                                                    "hour", "month", "timesincecasestart",
                                                    "timesincelastevent",
                                                    "timesincemidnight", "weekday"]},
        'SynLoan': {'TRACE': ['amount'],
                                       'EVENT': ["event_nr", "lifecycle:transition",
                                                 "hour", "month", "timesincecasestart",
                                                 "timesincelastevent",
                                                 "timesincemidnight", "weekday", "queue"]}
}

def read_log_csv(self, path):
    dataframe = pd.read_csv(path, sep=',')
    dataframe = pm4py.format_dataframe(dataframe, case_id='caseid', activity_key='task',
                                       timestamp_key='end_timestamp')
    event_log = pm4py.convert_to_event_log(dataframe)
    event_log = sorting.sort_timestamp(event_log, timestamp_key='start_timestamp')
    return event_log


def read_training(train, attrib_event, attrib_trace):
    # [event, event_processingTime, resource, wait, amount]
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
        attributes_trace = {}
        attributes_event = {}
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
    writer.writerow(['caseid', 'task', 'arrive:timestamp', 'start:timestamp', 'time:timestamp', 'role', 'open_cases', 'st_tsk_wip', 'queue'] +
                    ATTRIBUTES[NAME_EXPERIMENT]['EVENT'] + ['attrib_trace', 'label'])
    params.START_SIMULATION = arrivals[0][1]
    prev = arrivals[0][1]
    for i in range(0, len(arrivals)):
        next = arrivals[i][1]
        interval = (next - prev).total_seconds()
        prev = next
        yield env.timeout(interval)
        if str(arrivals[i][0]) in key:
            id_arrival = str(arrivals[i][0])
            env.process(Token(id_arrival, params, simulation_process, [], contrafactual[arrivals[i][0]].copy(), NAME_EXPERIMENT).simulation(env, writer, type))
        else:
            env.process(Token(arrivals[i][0], params, simulation_process, log[arrivals[i][0]].copy(), False, NAME_EXPERIMENT).simulation(env, writer, type))


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

def run_simulation(train_df, df_cf, NAME_EXPERIMENT, type ='rims', N_SIMULATION = 1):
    print(NAME_EXPERIMENT, N_SIMULATION, type)
    log, arrivals = read_training(train_df, ATTRIBUTES[NAME_EXPERIMENT]['EVENT'],
                                  ATTRIBUTES[NAME_EXPERIMENT]['TRACE'])
    contrafactual_traces, arrivals_CF = read_contrafactual(df_cf, ATTRIBUTES[NAME_EXPERIMENT]['EVENT'],
                                  ATTRIBUTES[NAME_EXPERIMENT]['TRACE'])
    arrivals = sorted(arrivals + arrivals_CF, key=lambda x: x[1])
    run(NAME_EXPERIMENT, type, log, arrivals, contrafactual_traces, list(contrafactual_traces.keys()))