

#### metrics to consider


### control-flow with CTD, 2-gram, 3-gram


### time perspective: CTD

from log_distance_measures.circadian_workforce_distribution import circadian_workforce_distribution_distance
from log_distance_measures.config import EventLogIDs
import pandas as pd
from datetime import datetime
from log_distance_measures.control_flow_log_distance import control_flow_log_distance
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance


def prefix_to_log(df, max_prefix):
    columns = ["caseid", "task", "start:timestamp", "time:timestamp", "user"]
    df_new = pd.DataFrame(columns=columns)

    # Ensure columns exist
    existing_columns = df.columns.tolist()

    for index, row in df.iterrows():
        case_id = row['trace_id']

        for i in range(1, max_prefix):
            prefix_col = f'prefix_{i}'
            start_col = f'start:timestamp_{i}'
            end_col = f'time:timestamp_{i}'
            resource_col = f'org:resource_{i}'

            # Check if column exists before accessing
            if prefix_col in existing_columns and row[prefix_col] != '0':
                start = datetime.utcfromtimestamp(row[start_col])
                end = datetime.utcfromtimestamp(row[end_col])
                row_data = {
                    "caseid": case_id,
                    "task": row[prefix_col],
                    "start:timestamp": start,
                    "time:timestamp": end,
                    "user": row[resource_col]
                }
                df_new = pd.concat([df_new, pd.DataFrame([row_data])], ignore_index=True)
    return df_new


# Set event log column ID mapping
test_log = EventLogIDs(  # These values are stored in DEFAULT_CSV_IDS
    case="caseid",
    activity="task",
    end_time="time:timestamp",
    start_time="start:timestamp",
    resource = 'user'
)

# Read and transform time attributes
original_log = pd.read_csv("/Users/francescameneghello/Desktop/RIMS_reproducibility/RIMS/ConsultaDataMining201618/results/rims/tst_ConsultaDataMining201618.csv")
original_log[test_log.start_time] = pd.to_datetime(original_log[test_log.start_time], utc=True)
original_log[test_log.end_time] = pd.to_datetime(original_log[test_log.end_time], utc=True)
method = ['SIMCED']
metrics = ['CLFD', '2-Gram', '3-Gram']
max_prefix = 10
path_dict = {
    'SIMCED': ['/Users/francescameneghello/Downloads/logs_consulta_bpic17_icpm/icpm_data_gen_eval/datasets/new_logs_icpm/ConsultaDataMining201618/ConsultaDataMining201618_test_df_cf_simulated_aug_0.2_pref_len_9.csv', test_log]
}

results = {}
for m in method:
    file = path_dict['SIMCED'][0]
    simulated_log = pd.read_csv(file)
    simulated_log = prefix_to_log(simulated_log, max_prefix)
    simulated_log[path_dict[m][1].start_time] = pd.to_datetime(simulated_log[path_dict[m][1].start_time], utc=True)
    simulated_log[path_dict[m][1].end_time] = pd.to_datetime(simulated_log[path_dict[m][1].end_time], utc=True)
    results[m] = {}
    for met in metrics:
        if met == 'CLFD':
           distance = control_flow_log_distance(
                original_log, test_log,  # First event log and its column id mappings
                simulated_log, path_dict[m][1]
           )
           print(met, distance)
        elif met == '2-Gram':
            distance = n_gram_distribution_distance(
                original_log, test_log,  # First event log and its column id mappings
                simulated_log, path_dict[m][1],
                n = 2
            )
            print(met, distance)
        else:
            distance = n_gram_distribution_distance(
                original_log, test_log,  # First event log and its column id mappings
                simulated_log, path_dict[m][1],
                n = 3
            )
            print(met, distance)

    results[m][met].append(distance)
    print(distance)