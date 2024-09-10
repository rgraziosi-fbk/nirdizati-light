import random
import math
import logging
import pm4py
import pandas as pd

logger = logging.getLogger(__name__)

def get_log(filepath: str, separator: str = ';'):
    """
    Reads a xes or csv log.

    For csv logs, standard column names must be used: 'case:concept:name' for the trace id, 'concept:name' for the activity, and 'time:timestamp' for the timestamp.

    Args:
        filepath (str): Path to the log.
        separator (str): In case of csv logs, the separator character used in the csv log.

    Returns:
        A pm4py EventLog object.
    """
    if filepath.endswith('.xes'):
        log = pm4py.read_xes(filepath)
    elif filepath.endswith('.csv'):
        log = pd.read_csv(filepath, sep=separator)
        log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])
    else:
        raise ValueError("Unsupported file extension")
    
    # ensure case id column is of type str
    log['case:concept:name'] = log['case:concept:name'].astype(str)
    
    return pm4py.convert_to_event_log(log, case_id_key='case:concept:name')


def split_train_val_test(
    log: pd.DataFrame,
    train_perc: float,
    val_perc: float,
    test_perc: float,
    shuffle: bool = False,
    seed: int = 42
):
    """
    Splits a DataFrame containing event log data into training, validation, and test sets.

    This function divides the DataFrame based on unique case identifiers (trace_id) into
    specified proportions for training, validation, and testing. It supports shuffling
    the cases before splitting to ensure random distribution.

    Args:
        log (pd.DataFrame): The input DataFrame containing the event log data.
        train_perc (float): The proportion of the data to be used for the training set.
        val_perc (float): The proportion of the data to be used for the validation set.
        test_perc (float): The proportion of the data to be used for the test set. This parameter
            is not directly used in splitting but can be useful for validation.
        shuffle (bool): If True, the cases are shuffled before splitting. Defaults to False.
        seed (int): The seed for the random number generator when shuffling. Defaults to 42.

    Returns:
        tuple: A tuple containing three pd.DataFrame objects for the training, validation, and test sets respectively.

    Raises:
        AssertionError: If the sum of the train, validation, and test percentage splits does not equal 1.
    """
    assert math.isclose(train_perc + val_perc + test_perc, 1, rel_tol=1e-9), "The sum of train_perc, val_perc, and test_perc should be equal to 1"

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
