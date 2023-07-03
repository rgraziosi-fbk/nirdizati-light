import glob
import os
import pathlib
import pandas as pd
import numpy as np
from pm4py.objects.log.exporter.xes.factory import export_log as export_log_xes
from pm4py.objects.conversion.log.factory import apply as df_to_eventLog
from pm4py.objects.log.util.general import CASE_ATTRIBUTE_PREFIX

from DatasetManager import DatasetManager

export_log = {
    '.xes': export_log_xes
}

input_path = '../results_eval'
input_path = '/Users/andrei/Desktop/PhD/nirdizati_light/results_eval/'
logs = [
  #  ('BPIC11_f1', 'BPIC11_f1'),
#    ('BPIC11_f2', 'BPIC11_f2'),
#    ('BPIC11_f3', 'BPIC11_f3'),
#    ('BPIC11_f4', 'BPIC11_f4'),
#    ('Production', 'Production'),
#     ('BPIC15_1_f2', 'BPIC15_1_f2'),
#    ('BPIC15_2_f2', 'BPIC15_2_f2'),
#    ('BPIC15_3_f2', 'BPIC15_3_f2'),
##    ('BPIC15_4_f2', 'BPIC15_4_f2'),
 #   ('BPIC15_5_f2', 'BPIC15_5_f2'),
#    ('BPIC17_O_Accepted', 'BPIC17_O_Accepted'),
#   ('BPIC17_O_Cancelled', 'BPIC17_O_Cancelled'),
#    ('BPIC17_O_Refused', 'BPIC17_O_Refused'),
    ('bpic2012_O_ACCEPTED-COMPLETE', 'bpic2012_O_ACCEPTED-COMPLETE'),
    ('bpic2012_O_CANCELLED-COMPLETE', 'bpic2012_O_CANCELLED-COMPLETE'),
  ('bpic2012_O_DECLINED-COMPLETE', 'bpic2012_O_DECLINED-COMPLETE'),
#    ('hospital_billing_2', 'hospital_billing_2'),
#    ('hospital_billing_3', 'hospital_billing_3'),
    ('sepsis_cases_1', 'sepsis_cases_1'),
    ('sepsis_cases_2', 'sepsis_cases_2'),
   ('sepsis_cases_4', 'sepsis_cases_4')
#    ('traffic_fines_1', 'traffic_fines_1'),
]
encodings = ['loreley']
#encodings=['complex']
#encodings=['complex','simptrace','loreley','simple']
#encodings = ['loreley']
prefix_length=[5,10,15,20,25]
desired_cfs = [5.0,10.0,15.0,20.0]
all_files = '/Users/andrei/Desktop/PhD/nirdizati_light/parsed_logs_loreley/'
export = '/Users/andrei/Desktop/PhD/nirdizati_light/parsed_logs_new_LORE/'
methods = ['balltree','kdtree','genetic','random']
#divide imported file by method, output method+dataset xes files for conformance checking
#simptrace_random_5 check sepsis simptrace random, could be empty
#sepsis 4 complex random 5
if __name__ == '__main__':
    for input_file, dataset_name in logs:
        for method in methods:
            for encoding in encodings:
                for prefix in prefix_length:
                    for k in desired_cfs:
                        file_path = os.path.join(all_files, ('parsed_dice_%s_%s_%s_preflen_%s_%s.csv' % (dataset_name,encoding,method,prefix,k)))
                        if os.path.exists(file_path) == True:
                            dataset_manager = DatasetManager(dataset_name,file_path)
                            data = dataset_manager.read_dataset()
                            cls_encoder_args = {'case_id_col': dataset_manager.case_id_col,
                                                'static_cat_cols': dataset_manager.static_cat_cols,
                                                'static_num_cols': dataset_manager.static_num_cols,
                                                'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                                                'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                                                'fillna': True}
                            #columns_to_rename = {'Case ID': 'case:concept:name'}

                            columns_to_rename = {'Case ID': 'case:concept:name'}
                            columns_to_rename.update({'Activity': 'concept:name'})
                            columns_to_rename.update({ c: CASE_ATTRIBUTE_PREFIX + c
                                for c in dataset_manager.static_cat_cols + dataset_manager.static_num_cols
                            })

                            data.rename(columns=columns_to_rename, inplace=True)

                            data['label'].replace({'regular': 'false', 'deviant': 'true'}, inplace=True)
                            data.replace('0','other',inplace=True)
                            if encoding == 'loreley':
                                data = data.replace({" \'": "","\'":""}, regex=True)
                            event_log = df_to_eventLog(data, parameters={
                                'return_variants': False
                            })

                            # if no 'Activity' is there but there is 'Activity Code' then use that
                            if 'concept:name' not in event_log[0][0] and 'Activity code' in event_log[0][0]:
                                columns_to_rename.update({'Activity code': 'concept:name'})
                                columns_to_rename.update({c: CASE_ATTRIBUTE_PREFIX + c
                                                          for c in dataset_manager.static_cat_cols + dataset_manager.static_num_cols
                                                          })

                                data.rename(columns=columns_to_rename, inplace=True)

                                data['label'].replace({'regular': 'false', 'deviant': 'true'}, inplace=True)

                                event_log = df_to_eventLog(data, parameters={
                                    'return_variants': False
                                })

                            attributes_to_promote = ['label']
                            for trace in event_log:
                                for attribute_to_promote in attributes_to_promote:
                                    if attribute_to_promote in trace[0]:
                                        trace.attributes[attribute_to_promote] = trace[0][attribute_to_promote]
                                        for event in trace:
                                            event.pop(attribute_to_promote)

                            export_log['.xes'](event_log, os.path.join(export, ('%s_dice_%s_%s_%s_%s.xes' % (dataset_name,encoding,method,prefix,k))))
                        else:
                            os.path.join(export, ('%s_dice_%s_%s_%s_%s.xes' % (dataset_name, encoding, method, prefix, k)))



