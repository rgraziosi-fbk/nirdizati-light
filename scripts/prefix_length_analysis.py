import os
import pathlib
from statistics import stdev, median, mode

from numpy import average, quantile
from pm4py.objects.log.importer.xes.factory import import_log as import_log_xes

import_log = {
    '.xes': import_log_xes
}
input_path = '../synthetic_data'

logs = [
    # 'dataset_test_integrazione_marrella/BPIC15_1_mean/',
    # 'sepsis_declare_labeling/',
    # 'dataset_test_integrazione_marrella/labelled_financial_log_mean/',
    # 'remake_forum_experiments/real/BPI11/f1/',
    # 'BPI15/f1/',
    # 'BPI12/f1/',
    'IRENE/BPIC11_f1',
    'IRENE/BPIC11_f2',
    'IRENE/BPIC11_f3',
    'IRENE/BPIC11_f4',
    'IRENE/BPIC15_1_f2',
    'IRENE/BPIC15_2_f2',
    'IRENE/BPIC15_3_f2',
    'IRENE/BPIC15_4_f2',
    'IRENE/BPIC15_5_f2',
    ### 'IRENE/BPIC17_O_Accepted',
    ### 'IRENE/BPIC17_O_Cancelled',
    ### 'IRENE/BPIC17_O_Refused',
    'IRENE/bpic2012_O_ACCEPTED-COMPLETE',
    'IRENE/bpic2012_O_CANCELLED-COMPLETE',
    'IRENE/bpic2012_O_DECLINED-COMPLETE',
    'IRENE/hospital_billing_2',
    'IRENE/hospital_billing_3',
    'IRENE/Production',  # available on ashkin
    'IRENE/sepsis_cases_1',
    'IRENE/sepsis_cases_2',
    'IRENE/sepsis_cases_4', # available on ashkin
    'IRENE/traffic_fines_1',
]


def get_log(filepath):
    # uses the xes, or csv importer depending on file type
    return import_log[pathlib.Path(filepath).suffixes[0]](filepath)


if __name__ == '__main__':
    print(
        'file_name', ', ',
        'min_len', ', ',
        'max_len', ', ',
        'average_len', ', ',
        'stdev_len', ', ',
        'median_len', ', ',
        'mode_len', ', ',
        'first_quartile_len', ', ',
        'second_quartile_len', ', ',
        'first_quintile_len', ', ',
        'second_quintile_len',
    )
    for input_file in logs:

        offset_path = os.path.join(input_path, input_file)
        event_log = get_log(os.path.join(offset_path, 'train.xes'))
        trace_lens = []
        for trace in event_log:
            trace_lens += [ len(trace) ]

        print(
            input_file, ', ',
            min(trace_lens), ', ',
            max(trace_lens), ', ',
            average(trace_lens), ', ',
            stdev(trace_lens), ', ',
            median(trace_lens), ', ',
            mode(trace_lens), ', ',
            quantile(trace_lens, .25), ', ',
            quantile(trace_lens, .50), ', ',
            quantile(trace_lens, .20), ', ',
            quantile(trace_lens, .40)
        )
