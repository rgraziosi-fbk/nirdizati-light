import pm4py
import numpy as np
import pandas as pd
import os
import DatasetManager
from datetime import datetime
''' At some point you will have to get the label from the dice output, keep that in mind going further, this is just
to test it out
Days and Months with 0 give eror to dateutil parser. Can change them to random values, or take them from the timestamp column
'''
input_path = '/Users/andrei/Desktop/PhD/nirdizati_light/results_cf_prefixes_encodings/'
export = '/Users/andrei/Desktop/PhD/nirdizati_light/parsed_logs_loreley/'
logs = [
#   ('BPIC11_f1', 'BPIC11_f1'),
#    ('BPIC11_f2', 'BPIC11_f2'),
#    ('BPIC11_f3', 'BPIC11_f3'),
#    ('BPIC11_f4', 'BPIC11_f4'),
#    ('Production', 'Production'),
#    ('BPIC15_1_f2', 'BPIC15_1_f2'),
#    ('BPIC15_2_f2', 'BPIC15_2_f2'),
#    ('BPIC15_3_f2', 'BPIC15_3_f2'),
#    ('BPIC15_4_f2', 'BPIC15_4_f2'),
#    ('BPIC15_5_f2', 'BPIC15_5_f2'),
#    ('BPIC17_O_Accepted', 'BPIC17_O_Accepted'),
#    ('BPIC17_O_Cancelled', 'BPIC17_O_Cancelled'),
#    ('BPIC17_O_Refused', 'BPIC17_O_Refused'),
 #   ('bpic2012_O_ACCEPTED-COMPLETE', 'bpic2012_O_ACCEPTED-COMPLETE'),
 #   ('bpic2012_O_CANCELLED-COMPLETE', 'bpic2012_O_CANCELLED-COMPLETE'),
  # ('bpic2012_O_DECLINED-COMPLETE', 'bpic2012_O_DECLINED-COMPLETE'),
#    ('hospital_billing_2', 'hospital_billing_2'),
#    ('hospital_billing_3', 'hospital_billing_3'),
    ('sepsis_cases_1', 'sepsis_cases_1'),
  #  ('sepsis_cases_2', 'sepsis_cases_2'),
 #   ('sepsis_cases_4', 'sepsis_cases_4')
#    ('traffic_fines_1', 'traffic_fines_1'),
]
nr_of_cfs = [5.0,10.0,15.0,20.0]
prefix_lengths = [5,10,15,20,25]
encodings = ['loreley']
methods = ['genetic']
if __name__ == '__main__':
    for input_file, dataset_name in logs:
        for encoding in encodings:
            for method in methods:
                for k in nr_of_cfs:
                    for prefix_length in prefix_lengths:
                        if encoding == 'complex':
                            dataset_loc = os.path.join(input_path, ('cf_%s_randomForestClassifier_dice_%s_%s_%s.csv' % (dataset_name,encoding, method,prefix_length)))
                            df = pd.read_csv(dataset_loc)
                            df = df.iloc[:,:-4]
                            df.insert(loc=0, column='Case ID', value=np.divmod(np.arange(len(df)), 1)[0] + 1)
                            df.insert(loc=1, column='label', value=1)
                            dataset_manager = DatasetManager(dataset_name,
                                                             input_file=os.path.join(input_path, ('cf_%s_randomForestClassifier_dice_%s_%s_%s.csv' % (dataset_name,encoding, method,prefix_length))))
                            static_columns = [dataset_manager.case_id_col] + dataset_manager.static_cat_cols + \
                                             [dataset_manager.label_col] + dataset_manager.static_num_cols
                            dynamic_columns = dataset_manager.dynamic_cat_cols+dataset_manager.dynamic_num_cols+[dataset_manager.timestamp_col]

                            for i in range(len(dynamic_columns)):
                                if dynamic_columns[i] == 'Activity':
                                    dynamic_columns[i] = 'prefix'
                                elif dynamic_columns[i] == 'Activity Code':
                                    dynamic_columns[i] = 'prefix'
                                elif dynamic_columns[i] == 'Activity code':
                                    dynamic_columns[i] = 'prefix'
                            long_data = pd.wide_to_long(df, stubnames=dynamic_columns, i='Case ID',
                                                        j='order', sep='_', suffix=r'\w+')
                            long_data_sorted = long_data.sort_values(['Case ID', 'order'], ).reset_index(drop=False)
                            for value in long_data_sorted[dataset_manager.timestamp_col][
                                long_data_sorted[dataset_manager.timestamp_col].map(type) == float]:
                                long_data_sorted[dataset_manager.timestamp_col].replace(value, datetime.fromtimestamp(value),
                                                                                        inplace=True)
        #                #timestamp to datetime
                        #save each file and iterate until the end, keep name convention
                        elif encoding == 'simple':
                            dataset_loc = os.path.join(input_path, ('cf_%s_randomForestClassifier_dice_%s_%s_%s.csv' %
                                                                     (dataset_name,encoding, method,prefix_length)))
                            df = pd.read_csv(dataset_loc)
                            df = df.iloc[:,:-4]
                            df.insert(loc=0, column='Case ID', value=np.divmod(np.arange(len(df)), 1)[0] + 1)
                            df.insert(loc=1, column='label', value=1)
                            dataset_manager = DatasetManager(dataset_name,
                                                             os.path.join(input_path,
                                                                          'cf_%s_randomForestClassifier_dice_%s_%s_%s.csv' %
                                                                     (dataset_name,encoding, method,prefix_length)))
                            dynamic_column = []
                            if dataset_manager.activity_col == 'Activity':
                                dynamic_column.append('prefix')
                            elif dataset_manager.activity_col == 'Activity Code':
                                dynamic_column.append('prefix')
                            elif dataset_manager.activity_col == 'Activity code':
                                dynamic_column.append('prefix')
                            long_data = pd.wide_to_long(df, stubnames=dynamic_column, i='Case ID',
                                                        j='order', sep='_', suffix=r'\w+')
                            timestamps = pd.date_range('1/1/2011', periods=len(long_data), freq='H')

                            long_data_sorted = long_data.sort_values(['Case ID', 'order'], ).reset_index(drop=False)
                            long_data_sorted[dataset_manager.timestamp_col] = timestamps
                        elif encoding == 'simptrace':
                            dataset_loc = os.path.join(input_path, ('cf_%s_randomForestClassifier_dice_%s_%s_%s.csv' %
                                                                    (dataset_name, encoding, method, prefix_length)))
                            df = pd.read_csv(dataset_loc)
                            df = df.iloc[:,:-4]
                            df.insert(loc=0, column='Case ID', value=np.divmod(np.arange(len(df)), 1)[0] + 1)
                            df.insert(loc=1, column='label', value=1)
                            dataset_manager = DatasetManager(dataset_name,
                                                             os.path.join(input_path,
                                                                          'cf_%s_randomForestClassifier_dice_%s_%s_%s.csv' %
                                                                          (dataset_name, encoding, method, prefix_length)))
                            static_columns = [dataset_manager.case_id_col] + dataset_manager.static_cat_cols + \
                                             [dataset_manager.label_col] + dataset_manager.static_num_cols
                            dynamic_column = []
                            if dataset_manager.activity_col == 'Activity':
                                dynamic_column.append('prefix')
                            elif dataset_manager.activity_col == 'Activity Code':
                                dynamic_column.append('prefix')
                            elif dataset_manager.activity_col == 'Activity code':
                                dynamic_column.append('prefix')
                            long_data = pd.wide_to_long(df, stubnames=dynamic_column, i='Case ID',
                                                        j='order', sep='_', suffix=r'\w+')
                            timestamps = pd.date_range('1/1/2011', periods=len(long_data), freq='H')

                            long_data_sorted = long_data.sort_values(['Case ID', 'order'], ).reset_index(drop=False)
                            long_data_sorted[dataset_manager.timestamp_col] = timestamps

                        else:
                            dataset_loc = os.path.join(input_path, ('cf_%s_randomForestClassifier_dice_%s_%s_%s.csv' %
                                                                    (dataset_name, encoding, method, prefix_length)))
                            df = pd.read_csv(dataset_loc)
                            if 'bpic2012' not in dataset_name:
                                df = df.iloc[:,:-4]
                            columns = ['prefix_'+str(i+1) for i in range(prefix_length)]
                            df = df[df['desired_cfs'] == k]
                            df = df.replace('0', pd.np.nan).dropna(axis=0, how='all', subset=columns).fillna(0).astype(
                                str)

                            df.insert(loc=0, column='Case ID', value=np.divmod(np.arange(len(df)), 1)[0] + 1)
                            df.insert(loc=1, column='label', value=1)
                            dataset_manager = DatasetManager(dataset_name,
                                                             os.path.join(input_path,
                                                                          'cf_%s_randomForestClassifier_dice_%s_%s_%s.csv' %
                                                                          (dataset_name, encoding, method, prefix_length)))
                            static_columns = [dataset_manager.case_id_col] + dataset_manager.static_cat_cols + \
                                             [dataset_manager.label_col] + dataset_manager.static_num_cols
                            dynamic_column = []
                            if dataset_manager.activity_col == 'Activity':
                                dynamic_column.append('prefix')
                            elif dataset_manager.activity_col == 'Activity Code':
                                dynamic_column.append('prefix')
                            elif dataset_manager.activity_col == 'Activity code':
                                dynamic_column.append('prefix')
                            long_data = pd.wide_to_long(df, stubnames=dynamic_column, i='Case ID',
                                                        j='order', sep='_', suffix=r'\w+')
                            timestamps = pd.date_range('1/1/2011', periods=len(long_data), freq='H')

                            long_data_sorted = long_data.sort_values(['Case ID', 'order'], ).reset_index(drop=False)
                            long_data_sorted[dataset_manager.timestamp_col] = timestamps

                        long_data_sorted['label'].replace({1:'regular'},inplace=True)
                        long_data_sorted.drop(columns=['order'],inplace=True)
                        long_data_sorted.rename(columns={'prefix':dataset_manager.activity_col},inplace=True)
                        long_data_sorted.to_csv(path_or_buf=os.path.join(export+'/parsed_dice_%s_%s_%s_preflen_%s_%s.csv'
                                                                         %
                                                                         (dataset_name,encoding,method,prefix_length,k))
                                                ,index=False)

