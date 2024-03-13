import logging
import pm4py
import pandas as pd

logger = logging.getLogger(__name__)

def get_log(filepath, separator=';'):
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
    
    return pm4py.convert_to_event_log(log)
