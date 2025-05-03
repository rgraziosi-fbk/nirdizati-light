from datetime import datetime
import csv
import simpy
from process import SimulationProcess
from event_trace import Token
from parameters import Parameters
import pandas as pd
from inter_trigger_timer import InterTriggerTimer
from datetime import timedelta
from utility import *
from itertools import groupby
from operator import itemgetter

PARALLEL = ['LacticAcid', 'CRP', 'Leucocytes', 'IV Liquid']

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
        'ConsultaDataMining201618': {'TRACE': [],
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
                                       'EVENT': [#"event_nr",
                                                 #"lifecycle:transition",
                                                 #"hour", "month", "timesincecasestart",
                                                 #"timesincelastevent",
                                                 #"timesincemidnight", "weekday", "queue"
                    ]}
}

TRACE_ATTRIBUTES = ['InfectionSuspected',
       'DiagnosticBlood', 'DisfuncOrg', 'SIRSCritTachypnea', 'Hypotensie',
       'SIRSCritHeartRate', 'Infusion', 'DiagnosticArtAstrup', 'Age',
       'DiagnosticIC', 'DiagnosticSputum', 'DiagnosticLiquor',
       'DiagnosticOther', 'SIRSCriteria2OrMore', 'DiagnosticXthorax',
       'SIRSCritTemperature', 'DiagnosticUrinaryCulture', 'SIRSCritLeucos',
       'Oligurie', 'DiagnosticLacticAcid', 'Diagnose', 'Hypoxie',
       'DiagnosticUrinarySediment', 'DiagnosticECG']
EVENT_ATTRIBUTES = ['Leucocytes', 'CRP', 'LacticAcid']



''' old version 
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
            processing = (end - start).total_seconds()
            if count_prefix <= 1:
                wait = 0
            else:
                start = row['start:timestamp_' + str(count_prefix)]
                end = row['time:timestamp_'+str(count_prefix-1)]
                wait = (end - start).total_seconds()
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
    return contrafactual_traces, arrivals_CF'''


def run_simulation(train_df, df_cf, NAME_EXPERIMENT, type ='rims', N_SIMULATION = 1):
    print(NAME_EXPERIMENT, N_SIMULATION, type)
    log, arrivals = read_training(train_df, ATTRIBUTES[NAME_EXPERIMENT]['EVENT'],
                                  ATTRIBUTES[NAME_EXPERIMENT]['TRACE'])
    contrafactual_traces, arrivals_CF = read_contrafactual(df_cf, ATTRIBUTES[NAME_EXPERIMENT]['EVENT'],
                                  ATTRIBUTES[NAME_EXPERIMENT]['TRACE'])
    arrivals = sorted(arrivals + arrivals_CF, key=lambda x: x[1])
    run(NAME_EXPERIMENT, type, log, arrivals, contrafactual_traces, list(contrafactual_traces.keys()))

def read_training(train):
    caseid_unique = list(train['caseid'].unique())
    traces = dict()
    for caseid in caseid_unique:
        trace = []
        event_attrib = {}
        trace_attrib = {}
        group_case = train[train['caseid'] == caseid]
        group_case = group_case.reset_index(drop=True)
        #### find parallel activities
        # Find duplicate available_time rows
        duplicates = group_case[group_case["available_time"].duplicated(keep=False)]
        groups = duplicates.groupby("available_time").groups
        parallel = list(groups.values())
        for index, row in group_case.iterrows():
            ### event: sequence/parallel, task, processing_time, resource, wait, event_attrib, event_event
            event = []
            event.append(False)
            event += [row['concept:name'], row['processing_time'], row['org:resource']]
            wait = (row['start:timestamp']-row["available_time"]).total_seconds()
            event.append(wait)
            for e in EVENT_ATTRIBUTES:
                event_attrib[e] = row[e]
            for t in TRACE_ATTRIBUTES:
                trace_attrib[t] = row[t]
            event.append(event_attrib)
            event.append(trace_attrib)
            ### find_parallel
            target_indices = group_case[group_case['concept:name'].isin(PARALLEL)].index.tolist()
            # Group sequences
            groups = []
            for k, g in groupby(enumerate(target_indices), lambda x: x[0] - x[1]):
                group = list(map(itemgetter(1), g))
                if len(group) >= 2:  # Only keep groups with 2 or more target activities
                    groups.append(group)
            in_parallel = next((i for i, sublist in enumerate(groups) if index in sublist), -1)
            if in_parallel > -1:
                first_event_in_parallel = groups[in_parallel][0]
                wait = (row['available_time']-group_case.iloc[first_event_in_parallel - 1]["time:timestamp"]).total_seconds()
                event[4] = wait
                if len(trace) > first_event_in_parallel:
                    trace[first_event_in_parallel][-1].append(event)
                else:
                    event[0] = True  ### there is a parallel
                    event.append([])
                    trace.append(event)
            else:
                trace.append(event)
        traces[caseid] = trace
    return traces




def setup(env: simpy.Environment, NAME_EXPERIMENT, params, i, type, traces_train, arrivals, contrafactual, key):
    simulation_process = SimulationProcess(env=env, params=params)
    path_result = 'simulated_log_' + NAME_EXPERIMENT + '_.csv'
    buffer_definition = { "id_case": -1, "activity": None, "role": None, "enabled_time": None, "start_time": None, "end_time": None, "resource": None, "prefix": Prefix}
    buffer_definition = buffer_definition | {a: None for a in EVENT_ATTRIBUTES} | {a: None for a in TRACE_ATTRIBUTES}
    print(buffer_definition)
    f = open(path_result, 'w')
    writer = csv.writer(f)
    writer.writerow(buffer_definition.keys())
    interval = InterTriggerTimer(params, simulation_process, params.START_SIMULATION)
    contrafactual = False
    for key in traces_train: ### to add also the contrafactual
        prefix = Prefix()
        itime = interval.get_next_arrival(env, i)
        yield env.timeout(0)
        parallel_object = ParallelObject()
        time_trace = params.START_SIMULATION + timedelta(seconds=env.now)
        env.process(
            Token(key, params, simulation_process, prefix, 'sequential', writer, parallel_object, time_trace,
                  traces_train[key], contrafactual, NAME_EXPERIMENT, buffer_definition, None).simulation(env))

def run(NAME_EXPERIMENT, log, arrivals, contrafactual, key):
    N_SIMULATION = 1
    N_TRACES = 1 #len(log)
    path_parameters = 'input_sepsis.json'
    for i in range(0, N_SIMULATION):
        params = Parameters(path_parameters, N_TRACES)
        env = simpy.Environment()
        env.process(setup(env, NAME_EXPERIMENT, params, i, type, log, arrivals, contrafactual, key))
        env.run(until=params.SIM_TIME)

def run_simulation_sepsis(train_df, df_cf, NAME_EXPERIMENT, N_SIMULATION=1):
    print(NAME_EXPERIMENT, N_SIMULATION, type)
    input_train = pd.read_csv('sepsis_start_test.csv', sep=",")
    input_train = input_train[input_train['caseid'] == 'AA']
    input_train['time:timestamp'] = pd.to_datetime(input_train['time:timestamp'])
    input_train['start:timestamp'] = pd.to_datetime(input_train['start:timestamp'])
    input_train['available_time'] = pd.to_datetime(input_train['available_time'])
    traces_train = read_training(input_train)
    run(NAME_EXPERIMENT, traces_train, None, [], [])


run_simulation_sepsis(None, None, 'SEPSIS', N_SIMULATION=1)