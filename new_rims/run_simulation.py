from datetime import datetime
import csv
import simpy
from new_rims.process import SimulationProcess
from new_rims.event_trace import Token
from new_rims.parameters import Parameters
import pandas as pd
from new_rims.inter_trigger_timer import InterTriggerTimer
from datetime import timedelta
from new_rims.utility import *
from itertools import groupby
from operator import itemgetter
import json

PARALLEL = ['LacticAcid', 'CRP', 'Leucocytes', 'IV Liquid',
            'Admission IC', 'Admission NC', 'ER Sepsis Triage', 'IV Antibiotics']

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
       'SIRSCritHeartRate', 'Infusion', 'DiagnosticArtAstrup',
       'DiagnosticIC', 'DiagnosticSputum', 'DiagnosticLiquor',
       'DiagnosticOther', 'SIRSCriteria2OrMore', 'DiagnosticXthorax',
       'SIRSCritTemperature', 'DiagnosticUrinaryCulture', 'SIRSCritLeucos',
       'Oligurie', 'DiagnosticLacticAcid', 'Diagnose', 'Hypoxie',
       'DiagnosticUrinarySediment', 'DiagnosticECG']
EVENT_ATTRIBUTES = ['Leucocytes', 'CRP', 'LacticAcid']

def find_parallel(row):
    prefix_trace = []
    for index in range(1, 25):
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
    parallel_find = [[x + 1 for x in sublist] for sublist in parallel_find]
    return parallel_find
def read_training(train, TRACE_ATTRIBUTES, EVENT_ATTRIBUTES):
    ### event = [sequence/parallel, task, processing_time, resource, wait, 'label', attributes_event, attributes_trace
    resource = 'Resource_'
    columns = list(train.columns)
    count_prefix = 1
    traces = dict()
    for index, row in train.iterrows():
        parallel_find = find_parallel(row)
        key = str(row['trace_id'])
        traces[key] = []
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
                #, task, processing_time, resource, wait
                if head_of_parallel == count_prefix:
                    traces[key].append(
                        [True, row[prefix], row['duration_' + str(count_prefix)], row[resource + str(count_prefix)], row['waiting_' + str(count_prefix)], row['label'], attributes_event, attributes_trace, []])
                    post_last_parallel = len(traces[key])-1
                else:
                    event = [False, row[prefix], row['duration_' + str(count_prefix)], row[resource + str(count_prefix)], row['waiting_' + str(count_prefix)], row['label'], attributes_event, attributes_trace]
                    traces[key][post_last_parallel][-1].append(event)
            else:
                traces[key].append(
                    [False, row[prefix], row['duration_' + str(count_prefix)], row[resource + str(count_prefix)], row['waiting_' + str(count_prefix)], row['label'], attributes_event, attributes_trace])
            count_prefix += 1
            prefix = 'prefix_' + str(count_prefix)
        count_prefix = 1

    return traces


def read_CF(contrafactual, TRACE_ATTRIBUTES, EVENT_ATTRIBUTES):
    ### event = [sequence/parallel, sequence/parallel, task, processing_time, resource, wait, 'label', attributes_event, attributes_trace
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
                        [True, row[prefix], -1, row[resource + str(count_prefix)], -1, row['label'], attributes_event, attributes_trace, []])
                    post_last_parallel = len(contrafactual_traces[key])-1
                else:
                    event = [False, row[prefix], -1, row[resource + str(count_prefix)], -1, row['label'], attributes_event, attributes_trace]
                    contrafactual_traces[key][post_last_parallel][-1].append(event)
            else:
                contrafactual_traces[key].append(
                    [False, row[prefix], -1, row[resource + str(count_prefix)], -1, row['label'], attributes_event, attributes_trace])
            count_prefix += 1
            prefix = 'prefix_' + str(count_prefix)
        count_prefix = 1

    return contrafactual_traces
def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z

def setup(env: simpy.Environment, NAME_EXPERIMENT, params, i, traces_train, traces_contrafactual, imbalance_factor, path_result):
    simulation_process = SimulationProcess(env=env, params=params)
    buffer_definition = { "id_case": -1, "activity": None, "role": None, "enabled_time": None, "start_time": None, "end_time": None, "resource": None, "prefix": Prefix}
    buffer_definition = buffer_definition | {a: None for a in EVENT_ATTRIBUTES} | {a: None for a in TRACE_ATTRIBUTES}
    print(buffer_definition)
    f = open(path_result, 'w')
    writer = csv.writer(f)
    writer.writerow(buffer_definition.keys())
    interval = InterTriggerTimer(params, simulation_process, params.START_SIMULATION)
    contrafactual = True
    traces = merge_two_dicts(traces_train, traces_contrafactual)
    for key in traces:
        prefix = Prefix()
        itime = interval.get_next_arrival(env, i)
        yield env.timeout(itime)
        parallel_object = ParallelObject()
        time_trace = params.START_SIMULATION + timedelta(seconds=env.now)
        if key not in traces_contrafactual:
            contrafactual = False
            env.process(
                Token(key, params, simulation_process, prefix, 'sequential', writer, parallel_object, time_trace,
                  traces[key], NAME_EXPERIMENT, buffer_definition, contrafactual).simulation(env))
        elif key in traces_contrafactual:
            contrafactual = True
            env.process(
                Token(key, params, simulation_process, prefix, 'sequential', writer, parallel_object, time_trace,
                  traces[key], NAME_EXPERIMENT, buffer_definition, contrafactual).simulation(env))

def run_simulation(train_df, df_cf, NAME_EXPERIMENT, imbalance_factor, path_result):
    print(NAME_EXPERIMENT)
    path_parameters = 'datasets/sepsis/input_sepsis.json'
    with open(path_parameters, 'r') as f:
        data = json.load(f)
        TRACE_ATTRIBUTES = data['TRACE_ATTRIBUTES']
        EVENT_ATTRIBUTES = data['EVENT_ATTRIBUTES']
    contrafactual_traces = read_CF(df_cf, TRACE_ATTRIBUTES, EVENT_ATTRIBUTES)
    #train_df = train_df.iloc[:10, :]
    train_traces = read_training(train_df,TRACE_ATTRIBUTES, EVENT_ATTRIBUTES)
    log = None
    N_TRACES = len(contrafactual_traces) + len(train_traces)
    N_SIMULATION = 1
    for i in range(0, N_SIMULATION):
        params = Parameters(path_parameters, N_TRACES)
        env = simpy.Environment()
        env.process(setup(env, NAME_EXPERIMENT, params, i, train_traces, contrafactual_traces, imbalance_factor, path_result))
        env.run()


NAME_EXPERIMENT = 'sepsis'
reconstructed_train_val_log_df = pd.read_csv('../datasets/sepsis/sepsis_df_cf_0.2_pref_len_25.csv')
reconstructed_df_cf = pd.read_csv('../datasets/sepsis/sepsis_train_df_0.2_pref_len_25.csv')
run_simulation(reconstructed_train_val_log_df, reconstructed_df_cf, NAME_EXPERIMENT, 0.2, '../datasets/sepsis/simulation.csv')