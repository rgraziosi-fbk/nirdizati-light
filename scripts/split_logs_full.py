import os
import pathlib

from pm4py.objects.log.exporter.xes.factory import export_log as export_log_xes
from pm4py.objects.log.importer.xes.factory import import_log as import_log_xes
from pm4py.objects.log.log import EventLog, Trace

import_log = {
    '.xes': import_log_xes
}

export_log = {
    '.xes': export_log_xes
}


def get_log(filepath):
    # uses the xes, or csv importer depending on file type
    return import_log[pathlib.Path(filepath).suffixes[0]](filepath)

input_path = '../'

logs = [
    # 'BPI15/f1',
    # 'BPI15/f2',
    # 'BPI15/f3'
    # 'BPI18/calculate_insert'
# 'dataset_test_integrazione_marrella/BPIC15_1_mean',
# 'dataset_test_integrazione_marrella/labelled_BPIC15_1_median',
# 'dataset_test_integrazione_marrella/labelled_financial_log_mean',
# 'dataset_test_integrazione_marrella/labelled_financial_log_median',
# 'dataset_test_integrazione_marrella/labelled_Sepsis Cases - Event Log_mean',
# 'dataset_test_integrazione_marrella/labelled_Sepsis Cases - Event Log_median'
# 'BPIC11_f1',
# 'BPIC11_f2',
# 'BPIC11_f3',
# 'BPIC11_f4',
# 'BPIC15_1_f2',
# 'BPIC15_2_f2',
#'BPIC15_3_f2',
#'BPIC15_4_f2',#
#'BPIC15_5_f2',
 'BPIC17_O_Accepted',
 'BPIC17_O_Cancelled',
 'BPIC17_O_Refused',
# 'bpic2012_O_ACCEPTED-COMPLETE',
# 'bpic2012_O_CANCELLED-COMPLETE',
# 'bpic2012_O_DECLINED-COMPLETE',
# 'hospital_billing_2',
# 'hospital_billing_3',
# 'Production',
# 'sepsis_cases_1',
# 'sepsis_cases_2',
# 'sepsis_cases_4',
# 'traffic_fines_1',

    # 'DONADELLO_KR/sepsis_95',
    # 'DONADELLO_KR/xray',

]

output_files = [
     ('train.xes', 64),
    #('train.xes', 48),
    ('validate.xes', 10),
    ('feedback.xes', 10),
    ('test.xes', 16)
]
'''
output_files = [
     ('0-33.xes', 33),
    #('train.xes', 48),
    ('33-66.xes', 33),
    ('66-full.xes', 34),
]
#output_files_process_full = [('full.xes',100)]
'''
if __name__ == '__main__':
    for input_file in logs:
        offset_path = os.path.join(input_path, input_file)
        event_log = get_log(os.path.join(offset_path, input_file+'.xes'))
        print('loaded =>', offset_path)
        print('len =>', len(event_log))

        offset = 0
        #for file, size in output_files:
        for file,size in output_files:
            current_split = EventLog()
            current_split._attributes = event_log.attributes
            current_split._attributes.pop('creator', None)
            current_split._attributes.pop('library', None)
            for index in range(int(len(event_log) * size / 100)):
                trace = event_log[index + offset]
                current_trace = Trace(attributes=trace.attributes)
                current_trace.attributes.pop('variant', None)
                current_trace.attributes.pop('variant-index', None)
                current_trace.attributes.pop('creator', None)
                if 'label' not in current_trace.attributes and 'label' in trace[0]:
                    current_trace.attributes['label'] = trace[0]['label']
                for event in trace:
                    event._dict.pop('Variant', None)
                    event._dict.pop('Variant_index', None)
                    event._dict.pop('label', None)
                    if 'case:concept:name' in event:
                        activity_name = 'case:concept:name'
                    elif 'concept:name' in event:
                        activity_name = 'concept:name'
                    elif 'Activity' in event:
                        activity_name = 'Activity'
                    elif 'Activity code' in event:
                        activity_name = 'Activity code'
                    else:
                        raise Exception('can not find activity name')
                    event['concept:name'] = event[activity_name].lower()\
                        .replace(' ', '')\
                        .replace('-', '')\
                        .replace(':', '')\
                        .replace('.', '')\
                        .replace(',', '')\
                        .replace('&', '')\
                        .replace('_', '')
                    if activity_name != 'concept:name':
                        event._dict.pop(activity_name, None)
                    current_trace.append(event)
                current_split.append(current_trace)

            offset += int(len(event_log) * size / 100)

            export_log['.xes'](current_split, os.path.join(offset_path, file))

            print('file', file, 'len', len(current_split))
