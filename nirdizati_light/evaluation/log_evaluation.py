from log_distance_measures.circadian_event_distribution import circadian_event_distribution_distance
from log_distance_measures.case_arrival_distribution import case_arrival_distribution_distance
from log_distance_measures.config import AbsoluteTimestampType, DEFAULT_CSV_IDS
import math
from pm4py.objects.log.util import dataframe_utils
from pm4py import convert_to_event_log
from log_distance_measures.config import EventLogIDs, AbsoluteTimestampType, discretize_to_hour,discretize_to_day
from log_distance_measures.control_flow_log_distance import control_flow_log_distance
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance
from log_distance_measures.case_arrival_distribution import case_arrival_distribution_distance
from log_distance_measures.circadian_event_distribution import circadian_event_distribution_distance
from log_distance_measures.relative_event_distribution import relative_event_distribution_distance
from log_distance_measures.work_in_progress import work_in_progress_distance
from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance
from log_distance_measures.circadian_workforce_distribution import circadian_workforce_distribution_distance
import pandas as pd
from declare4py.declare4py import Declare4Py
from declare4py.enums import TraceState
import numpy as np
def conformance_score(simulated_log, CONF, dataset, model_path):
    d4py = Declare4Py()
    d4py.parse_decl_model(model_path=model_path+dataset+'_'+str(CONF['prefix_length'])+'.decl')
    simulated_log.rename({'start_timestamp':'start:timestamp', 'time_timestamp': 'time:timestamp','Case ID':'case:concept:name','Activity':'concept:name'}, axis=1, inplace=True)
    simulated_log = convert_to_event_log(simulated_log)
    d4py.load_xes_log(simulated_log)
    model_check_sim = d4py.conformance_checking(consider_vacuity=False)
    model_check_simulated = {
        constraint
        for trace, patts in model_check_sim.items()
        for constraint, checker in patts.items()
        if checker.state == TraceState.SATISFIED
    }

    model_check_simulated = {
        trace: {
            constraint: checker
            for constraint, checker in constraints.items()
            if checker.state == TraceState.SATISFIED
        }
        for trace, constraints in model_check_sim.items()
    }

    try:
        conformance_score = [len(v) / len(d4py.model.constraints) for v in model_check_simulated.values()]
    except ZeroDivisionError:
        print('No constraints available')
        conformance_score = [1.0]
    avg_conformance = np.mean(conformance_score)
    print('Average conformance score', np.mean(conformance_score))
    return avg_conformance
def start_error(log):
    error = 0
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])
    log['start:timestamp'] = pd.to_datetime(log['start:timestamp'])
    for index, row in log.iterrows():
        if row['start:timestamp']>row['time:timestamp']:
            error += 1
    return error
def overlap_error(log):
    log = log[log['Resource'].notna()]
    res = log['Resource'].unique().tolist()
    capacity = {r: 1 for r in res}
    overlap_error = 0
    resource_work = {r: {} for r in res}
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])
    log['start:timestamp'] = pd.to_datetime(log['start:timestamp'])
    for index, row in log.iterrows():
        start = int(row['start:timestamp'].timestamp())
        end = int(row['time:timestamp'].timestamp())
        for i in range(start, end):
            if i in resource_work[row['Resource']]:
                resource_work[row['Resource']][i] += 1
                if resource_work[row['Resource']][i] > capacity[row['Resource']]:
                    overlap_error += 1
            else:
                resource_work[row['Resource']][i] = 1
    return overlap_error
def logs_evaluation(original_log, generated_log, timestamp_col_name=None, csv_ids=None,
                    absolute_timestamp_type=AbsoluteTimestampType.END, discretize_to_day=discretize_to_day, CONF=None,
                    dataset=None, d4py=None, model_path=None):
    """
    Evaluate the generated log against the original log using various distance measures.
    :param original_log: The original log to compare against.
    :param generated_log: The generated log to evaluate.
    :param timestamp_col_name: The name of the timestamp column in the logs.
    :param csv_ids: The IDs of the CSV files used for evaluation.
    :param absolute_timestamp_type: The type of absolute timestamp to use for evaluation.
    :param discretize_to_day: Whether to discretize the timestamps to days.
    :return: A dictionary containing the results of the evaluation.
    """
    if 'time:timestamp' in original_log.columns:
        original_log.rename(columns={'time:timestamp': 'time_timestamp'}, inplace=True)
    if 'time:timestamp' in generated_log.columns:
        generated_log.rename(columns={'time:timestamp': 'time_timestamp'}, inplace=True)
    if 'start:timestamp' in original_log.columns:
        original_log.rename(columns={'start:timestamp': 'start_timestamp'}, inplace=True)
    if 'start:timestamp' in generated_log.columns:
        generated_log.rename(columns={'start:timestamp': 'start_timestamp'}, inplace=True)
    log_ids = EventLogIDs(activity = 'Activity',
                          case = 'Case ID',
                          start_time='start_timestamp',
                          end_time='time_timestamp',
                          resource='Resource')

    original_log.start_timestamp = pd.to_datetime(original_log.start_timestamp, utc=True)
    original_log.time_timestamp = pd.to_datetime(original_log.time_timestamp, utc=True)

    generated_log.start_timestamp = pd.to_datetime(generated_log.start_timestamp, utc=True)
    generated_log.time_timestamp = pd.to_datetime(generated_log.time_timestamp, utc=True)

    results = {
        'aed': absolute_event_distribution_distance(original_log, log_ids, generated_log, log_ids,
                                     AbsoluteTimestampType.BOTH,discretize_to_hour),
        'circadian_event_distribution_distance': absolute_event_distribution_distance(original_log,log_ids, generated_log, log_ids,
                                     AbsoluteTimestampType.BOTH,discretize_to_day),
        'case_arrival_distribution_distance': absolute_event_distribution_distance(original_log,log_ids, generated_log, log_ids,
                                     AbsoluteTimestampType.BOTH,discretize_to_hour),
        'cfld': control_flow_log_distance(original_log,
      log_ids,
      generated_log,
      log_ids,
      parallel=False),
        'red': relative_event_distribution_distance(
      original_log,
      log_ids,
      generated_log,
      log_ids,
      discretize_type=AbsoluteTimestampType.BOTH,
      discretize_event=discretize_to_hour,
    ),
        'ctd': cycle_time_distribution_distance(
      original_log,
      log_ids,
      generated_log,
      log_ids,
      bin_size=pd.Timedelta(hours=1),
    ),
        'cwd': circadian_workforce_distribution_distance(
      original_log,
      log_ids,
      generated_log,
      log_ids,

    ),
        'wipd': work_in_progress_distance(
      original_log,
      log_ids,
      generated_log,
      log_ids,
    ),
        'ngd': n_gram_distribution_distance(
      original_log,
      log_ids,
      generated_log,
      log_ids,
    ),
        'conformance_score': conformance_score(generated_log, CONF, dataset, model_path),
    'start_error':start_error(generated_log),
    'overlap_error': overlap_error(generated_log),
    }

    return results
