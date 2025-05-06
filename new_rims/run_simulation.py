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
import json

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

def find_parallel(row):
    prefix_trace = []
    for index in range(1, 20):
        prefix = 'prefix_' + str(index)
        prefix_trace.append(1 if row[prefix] in PARALLEL else 0)
    parallel_find = []
    index = 0
    open_sub = []
    while index<len(prefix_trace):
        if prefix_trace[index] == 1:
            if len(open_sub) == 0:
                open_sub = [index]
            else:
                open_sub.append(index)
        else:
            if len(open_sub) > 1:
                parallel_find.append(open_sub)
            open_sub = []
        index+=1
    if len(open_sub) > 0:
        parallel_find.append(open_sub)
    return parallel_find

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

def read_CF(contrafactual, TRACE_ATTRIBUTES, EVENT_ATTRIBUTES):
    resource = 'Resource_'
    columns = list(contrafactual.columns)
    count_prefix = 1
    contrafactual_traces = dict()
    for index, row in contrafactual.iterrows():
        parallel_find = find_parallel(row)
        key = str(row['trace_id']) + "_CF"
        contrafactual_traces[key] = []
        prefix = 'prefix_' + str(count_prefix)
        attributes_trace = {}
        attributes_event = {}
        for k in TRACE_ATTRIBUTES:
            if k in row:
                attributes_trace[k] = row[k]
        for k in EVENT_ATTRIBUTES:
            attributes_event[k] = row[k + '_' + str(count_prefix)]
        post_last_parallel = 0
        while prefix in columns and row[prefix] != '0' and row[prefix] != 0:
            index = next((i for i, sub in enumerate(parallel_find) if count_prefix in sub), -1)
            if index > -1:
                head_of_parallel = parallel_find[index][0]
                if head_of_parallel == count_prefix:
                    contrafactual_traces[key].append(
                        [True, row[prefix], row[resource + str(count_prefix)], row['label'], attributes_event, attributes_trace, []])
                    post_last_parallel = len(contrafactual_traces[key])-1
                else:
                    event = [False, row[prefix], row[resource + str(count_prefix)], row['label'], attributes_event, attributes_trace]
                    contrafactual_traces[key][post_last_parallel][-1].append(event)
            else:
                contrafactual_traces[key].append(
                    [False, row[prefix], row[resource + str(count_prefix)], row['label'], attributes_event, attributes_trace])
            count_prefix += 1
            prefix = 'prefix_' + str(count_prefix)
        count_prefix = 1

    return contrafactual_traces


def setup(env: simpy.Environment, NAME_EXPERIMENT, params, i, traces_train, traces_contrafactual):
    simulation_process = SimulationProcess(env=env, params=params)
    path_result = 'simulated_log_' + NAME_EXPERIMENT + '_.csv'
    buffer_definition = { "id_case": -1, "activity": None, "role": None, "enabled_time": None, "start_time": None, "end_time": None, "resource": None, "prefix": Prefix}
    buffer_definition = buffer_definition | {a: None for a in EVENT_ATTRIBUTES} | {a: None for a in TRACE_ATTRIBUTES}
    print(buffer_definition)
    f = open(path_result, 'w')
    writer = csv.writer(f)
    writer.writerow(buffer_definition.keys())
    interval = InterTriggerTimer(params, simulation_process, params.START_SIMULATION)
    contrafactual = True
    for key in traces_contrafactual: ### to add also the traces_train
        prefix = Prefix()
        itime = interval.get_next_arrival(env, i)
        yield env.timeout(itime)
        parallel_object = ParallelObject()
        time_trace = params.START_SIMULATION + timedelta(seconds=env.now)
        env.process(
            Token(key, params, simulation_process, prefix, 'sequential', writer, parallel_object, time_trace,
                  traces_contrafactual[key], NAME_EXPERIMENT, buffer_definition, contrafactual).simulation(env))


def run_simulation(train_df, df_cf, NAME_EXPERIMENT):
    print(NAME_EXPERIMENT)
    path_parameters = 'input_sepsis.json'
    with open(path_parameters, 'r') as f:
        data = json.load(f)
        TRACE_ATTRIBUTES = data['TRACE_ATTRIBUTES']
        EVENT_ATTRIBUTES = data['EVENT_ATTRIBUTES']
    contrafactual_traces = read_CF(df_cf, TRACE_ATTRIBUTES, EVENT_ATTRIBUTES)
    log = None
    N_TRACES = len(contrafactual_traces)
    N_SIMULATION = 1
    for i in range(0, N_SIMULATION):
        params = Parameters(path_parameters, N_TRACES)
        env = simpy.Environment()
        env.process(setup(env, NAME_EXPERIMENT, params, i, log, contrafactual_traces))
        env.run(until=params.SIM_TIME)

NAME_EXPERIMENT = 'sepsis'
df_cf = pd.read_csv('cfs.csv', sep=",")
run_simulation(None, df_cf, NAME_EXPERIMENT)


'''def run_simulation_sepsis(train_df, df_cf, NAME_EXPERIMENT, N_SIMULATION=1):
    print(NAME_EXPERIMENT, N_SIMULATION, type)
    input_train = pd.read_csv('sepsis_start_test.csv', sep=",")
    input_train = input_train[input_train['caseid'] == 'AA']
    input_train['time:timestamp'] = pd.to_datetime(input_train['time:timestamp'])
    input_train['start:timestamp'] = pd.to_datetime(input_train['start:timestamp'])
    input_train['available_time'] = pd.to_datetime(input_train['available_time'])
    traces_train = read_training(input_train)
    run(NAME_EXPERIMENT, traces_train, None, [], [])
run_simulation_sepsis(None, None, 'SEPSIS', N_SIMULATION=1)'''