import logging
import pathlib

from pm4py.objects.conversion.log import factory as conversion_factory
from pm4py.objects.log.exporter.csv.factory import export_log as export_log_csv
from pm4py.objects.log.exporter.xes.factory import export_log as export_log_xes
from pm4py.objects.log.importer.csv.factory import import_event_stream
from pm4py.objects.log.importer.xes.factory import import_log as import_log_xes
from pm4py.objects.log.log import EventLog
from pm4py.util import constants

logger = logging.getLogger(__name__)


def import_log_csv(path):
    return conversion_factory.apply(
        import_event_stream(path),                           # https://pm4py.fit.fraunhofer.de/documentation/1.2
        parameters={constants.PARAMETER_CONSTANT_CASEID_KEY: "case:concept:name",     # this tells the importer
                    constants.PARAMETER_CONSTANT_ACTIVITY_KEY: "concept:name",        # how to parse the csv
                    constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: "time:timestamp"}     # and which are the caseID
    )                                                                                 # concept name and timestamp


import_log = {
    '.csv': import_log_csv,
    '.xes': import_log_xes
}

export_log = {
    '.csv': export_log_csv,
    '.xes': export_log_xes
}


def get_log(filepath: str = None) -> EventLog:
    """Read in event log from disk

    Uses xes_importer to parse log.
    """
    logger.info("\t\tReading in log from {}".format(filepath))
    # uses the xes, or csv importer depending on file type
    return import_log[pathlib.Path(filepath).suffixes[0]](filepath)
