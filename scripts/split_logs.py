import os
import pathlib

from pm4py import read_xes,write_xes
from pm4py.objects.log.log import EventLog, Trace

#import_log = {# 
#    '.xes': import_log_xes
#}

#export_log = {
#    '.xes': export_log_xes
#}


def get_log(filepath):
    # uses the xes, or csv importer depending on file type
    return read_xes(filepath)

#input_path = '../data'

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
# 'IRENE/BPIC11_f1',
# 'IRENE/BPIC11_f2',
# 'IRENE/BPIC11_f3',
# 'IRENE/BPIC11_f4',
# 'IRENE/BPIC15_1_f2',
# 'IRENE/BPIC15_2_f2',
#'IRENE/BPIC15_3_f2',
#'IRENE/BPIC15_4_f2',
#'IRENE/BPIC15_5_f2',
 #'DATASETS_IRENE/BPIC17_O_Accepted',
 #'DATASETS_IRENE/BPIC17_O_Cancelled',
 #'DATASETS_IRENE/BPIC17_O_Refused',
 '../bpic2012_O_ACCEPTED-COMPLETE/',
 '../bpic2012_O_CANCELLED-COMPLETE/',
 '../bpic2012_O_DECLINED-COMPLETE/',
# 'IRENE/hospital_billing_2',
# 'IRENE/hospital_billing_3',
# 'IRENE/Production',
# 'IRENE/sepsis_cases_1',
# 'IRENE/sepsis_cases_2',
# 'IRENE/sepsis_cases_4',
# 'IRENE/traffic_fines_1',

    # 'DONADELLO_KR/sepsis_95',
    # 'DONADELLO_KR/xray'
    #'../synthetic_bank_accepted/',
    #'../synthetic_bank_cancelled/',
    #'../synthetic_bank_declined/',
]

output_files = [
    ('full.xes',100)
     #('train.xes', 73),
    #('train.xes', 48),
    #('validate.xes', 10),
    #('feedback.xes', 1),
    #('test.xes', 16)
]

#input_path =''
if __name__ == '__main__':
    for log in logs:
        offset_path = log
        event_log = get_log(offset_path+ 'full.xes')
        print('loaded =>', offset_path)
        print('len =>', len(event_log))

        offset = 0
        for file, size in output_files:
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
            write_xes(current_split, os.path.join(offset_path, file))

            print('file', file, 'len', len(current_split))
