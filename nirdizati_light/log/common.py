import logging

from pm4py import read_xes

logger = logging.getLogger(__name__)


def import_log_csv(path):
    dataframe = pd.read_csv(path,sep=',')
    dataframe = pm4py.format_dataframe(dataframe, case_id='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp')
    event_log = pm4py.convert_to_event_log(dataframe)
    return event_log




def get_log(filepath):
    """Read in event log from disk
    Uses xes_importer to parse log.
    """
    logger.info("\t\tReading in log from {}".format(filepath))
    # uses the xes, or csv importer depending on file type
    event_log = read_xes(filepath)
    return event_log
