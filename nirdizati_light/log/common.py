import logging
import pm4py
import pandas as pd
import random

logger = logging.getLogger(__name__)

def get_log(filepath: str, separator: str = ';', case_id_key: str = 'case:concept:name'):
    """
    Read a xes or csv log
    
    :param str filepath: path to the log
    :param str separator: in case of csv logs, the separator character used in the csv log

    :return: a pm4py EventLog object
    """
    if filepath.endswith('.xes'):
        log = pm4py.read_xes(filepath)
    elif filepath.endswith('.csv'):
        log = pd.read_csv(filepath, sep=separator)
        log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])
    else:
        raise ValueError("Unsupported file extension")
    
    # ensure case id column is of type str
    log[case_id_key] = log[case_id_key].astype(str)
    
    return pm4py.convert_to_event_log(log, case_id_key=case_id_key)


def split_train_val_test(log: pd.DataFrame, train_perc: float, val_perc: float, test_perc: float, shuffle: bool = False, seed: int = 42):
    cases = list(log['trace_id'].unique())

    if shuffle:
        random.seed(seed)
        random.shuffle(cases)

    train_size = int(train_perc * len(cases))
    val_size = int(val_perc * len(cases))

    train_cases = cases[:train_size]
    val_cases = cases[train_size:train_size + val_size]
    test_cases = cases[train_size + val_size:]

    assert len(train_cases) + len(val_cases) + len(test_cases) == len(cases)

    train_df = log[log['trace_id'].isin(train_cases)]
    val_df = log[log['trace_id'].isin(val_cases)]
    test_df = log[log['trace_id'].isin(test_cases)]

    return train_df, val_df, test_df
