import logging
import warnings
import os
import numpy as np
import pandas as pd
import pm4py
from sklearn.model_selection import train_test_split
from nirdizati_light.encoding.common import get_encoded_df, EncodingType
from nirdizati_light.encoding.constants import TaskGenerationType, PrefixLengthStrategy, EncodingTypeAttribute
from nirdizati_light.encoding.time_encoding import TimeEncodingType
from nirdizati_light.evaluation.common import evaluate_classifier,evaluate_regressor
from nirdizati_light.explanation.common import ExplainerType, explain
from nirdizati_light.hyperparameter_optimisation.common import retrieve_best_model, HyperoptTarget
from nirdizati_light.labeling.common import LabelTypes
from nirdizati_light.log.common import get_log
from nirdizati_light.predictive_model.common import ClassificationMethods, get_tensor, RegressionMethods
from nirdizati_light.predictive_model.predictive_model import PredictiveModel, drop_columns
from nirdizati_light.evaluation.log_evaluation import logs_evaluation
import random
import json
from pm4py import convert_to_event_log, write_xes
from dataset_confs import DatasetConfs
from new_rims.run_simulation import run_simulation
import ast
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
import itertools


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict

def convert_to_log(simulated_log, cols):
    simulated_log = pm4py.convert_to_event_log(simulated_log)
    for trace in simulated_log:
        for c in cols:
            trace.attributes[c] = trace[0][c]
            for e in trace:
                del e[c]
    pm4py.write_xes(simulated_log, 'exported.xes')
    return simulated_log

def run_simple_pipeline(CONF=None, dataset_name=None):
    random.seed(CONF['seed'])
    np.random.seed(CONF['seed'])
    dataset_confs = DatasetConfs(dataset_name=dataset_name, where_is_the_file=CONF['data'])

    logger.debug('LOAD DATA')
    log = get_log(filepath=CONF['data'])
    logger.debug('ENCODE DATA')
    encoder, full_df = get_encoded_df(log=log, CONF=CONF)

    #full_df = full_df[full_df.columns[~pd.Series(full_df.columns).str.contains(
    #    'cases|time|queue|open|group|event|lifecycle|day|hour|week|month')]]
    #encoder.decode(full_df)
    def reconstruct_timestamps(df):
        """Reconstruct time:timestamp, arrival:timestamp, and start:timestamp columns iteratively."""
        reconstructed_df = df.copy()# Avoid modifying the original DataFrame

        # Ensure start_trace exists and rename it to start:timestamp_1
        if "start_trace" in reconstructed_df.columns:
            reconstructed_df.rename(columns={"start_trace": "start:timestamp_1"}, inplace=True)

        # Convert start:timestamp_1 to datetime
        reconstructed_df["start:timestamp_1"] = pd.to_datetime(reconstructed_df["start:timestamp_1"], unit='s',
                                                               errors='coerce')
        reconstructed_df.insert(reconstructed_df.columns.get_loc('prefix_1') + 1, 'start:timestamp_1',
                                reconstructed_df.pop('start:timestamp_1'))

        for prefix in range(1, CONF['prefix_length'] + 1):
            prefix_col = f'prefix_{prefix}'
            # Start from 1
            duration_col = f'duration_{prefix}'
            waiting_col = f'waiting_{prefix}'
            arrival_col = f'arrival_{prefix}'
            start_timestamp_col = f'start:timestamp_{prefix}'
            time_timestamp_col = f'time:timestamp_{prefix}'
            if prefix < CONF['prefix_length']:
                next_start_timestamp_col = f'start:timestamp_{prefix + 1}'

            # Compute time:timestamp_x using start:timestamp_x + duration_x
            if duration_col in reconstructed_df.columns:
                mask = reconstructed_df[start_timestamp_col] != 0  # Ensure non-zero timestamps
                reconstructed_df.loc[mask, time_timestamp_col] = reconstructed_df.loc[
                                                                     mask, start_timestamp_col] + pd.to_timedelta(
                    reconstructed_df.loc[mask, duration_col], unit='s'
                )
                reconstructed_df.loc[~mask, time_timestamp_col] = 0  # If start_timestamp_col is 0, keep it 0
                # Insert time:timestamp_x **right after start:timestamp_x**
                idx = reconstructed_df.columns.get_loc(prefix_col)
                reconstructed_df.insert(idx + 2, time_timestamp_col, reconstructed_df.pop(time_timestamp_col))

            if waiting_col in reconstructed_df.columns and next_start_timestamp_col:
                mask = reconstructed_df[time_timestamp_col] != 0  # Ensure non-zero timestamps
                reconstructed_df[next_start_timestamp_col] = reconstructed_df.loc[mask, time_timestamp_col] + pd.to_timedelta(
                    reconstructed_df.loc[mask, arrival_col], unit='s'
                )
                reconstructed_df.loc[~mask, next_start_timestamp_col] = 0
                if prefix < CONF['prefix_length']:
                    idx_start = reconstructed_df.columns.get_loc(f'prefix_{prefix+1}')
                    reconstructed_df.insert(idx_start + 1, next_start_timestamp_col, reconstructed_df.pop(next_start_timestamp_col))
                # If time_timestamp_col is 0, keep it 0
            if prefix_col in reconstructed_df.columns:
                zero_mask = reconstructed_df[prefix_col] == '0'
                for col in [duration_col, waiting_col, arrival_col, start_timestamp_col, time_timestamp_col,next_start_timestamp_col]:
                        reconstructed_df.loc[zero_mask, col] = 0
        for prefix in range(1, CONF['prefix_length'] + 1):
            start_timestamp_col = f'start:timestamp_{prefix}'
            time_timestamp_col = f'time:timestamp_{prefix}'
            reconstructed_df[start_timestamp_col] = reconstructed_df[start_timestamp_col].apply(lambda x: x.timestamp() if x != 0 else 0)
            reconstructed_df[time_timestamp_col] = reconstructed_df[time_timestamp_col].apply(lambda x: x.timestamp() if x != 0 else 0)

        reconstructed_df = reconstructed_df[reconstructed_df.columns[~pd.Series(reconstructed_df.columns).str.contains(
            'arrival|waiting|duration')]]
        return reconstructed_df

    #reconstructed_df = reconstruct_timestamps(full_df)
    logger.debug('TRAIN PREDICTIVE MODEL')
    # split in train, val, test
    train_size = CONF['train_val_test_split'][0]
    val_size = CONF['train_val_test_split'][1]
    test_size = CONF['train_val_test_split'][2]
    if train_size + val_size + test_size != 1.0:
        raise Exception('Train-val-test split does  not sum up to 1')

    #full_df = full_df[full_df.columns.drop(list(full_df.filter(regex='Resource')))]
    #full_df = full_df[full_df.columns.drop(list(full_df.filter(regex='Activity')))]
    # Assume 'full_df' is your complete DataFrame, and 'target' is the column you're stratifying on
    X = full_df.drop('label', axis=1)  # Features (remove target column)
    y = full_df['label']  # Target column

    if CONF['label_to_gen'] == 'regular':
        y_min = encoder._label_dict['label']['regular']
        y_maj = encoder._label_dict['label']['deviant']
    elif CONF['label_to_gen'] == 'deviant':
        y_min = encoder._label_dict['label']['deviant']
        y_maj = encoder._label_dict['label']['regular']
    full_df_maj = full_df[full_df['label'] == y_maj]
    full_df_min = full_df[full_df['label'] == y_min]

    train_df_maj = full_df_maj
    # 20% from Label B
    train_df_min = full_df_min.sample(frac=CONF['undersampling_factor'], random_state=CONF['seed'])
    #Remove the 20%
    test_df = full_df_min.drop(train_df_min.index)

    train_df = pd.concat([train_df_maj, train_df_min])
    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']

    from sklearn.model_selection import StratifiedShuffleSplit,ShuffleSplit
    ss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=CONF['seed'])
    for train_index, val_index in ss_val_test.split(X_train, y_train):
        X_train, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

    # Now you have your splits
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    '''
    encoder.decode(train_df)
    encoder.decode(val_df)
    encoder.decode(test_df)

    reconstructed_train_df = reconstruct_timestamps(train_df)
    reconstructed_val_df = reconstruct_timestamps(val_df)
    reconstructed_test_df = reconstruct_timestamps(test_df)

    cols = [*dataset_confs.dynamic_num_cols.values(), *dataset_confs.dynamic_cat_cols.values()]
    event_cols = list(itertools.chain.from_iterable(cols))+ ['prefix','time:timestamp']
    long_train_df = pd.wide_to_long(reconstructed_train_df, stubnames=event_cols, i='trace_id',j='order', sep='_', suffix=r'\w+')
    long_val_df = pd.wide_to_long(reconstructed_val_df, stubnames=event_cols, i='trace_id',j='order', sep='_', suffix=r'\w+')
    long_test_df = pd.wide_to_long(reconstructed_test_df, stubnames=event_cols, i='trace_id',j='order', sep='_', suffix=r'\w+')
    long_train_df.reset_index(inplace=True)
    long_val_df.reset_index(inplace=True)
    long_test_df.reset_index(inplace=True)
    long_train_df.drop(columns=['order','concept:name'], inplace=True
                       )
    long_val_df.drop(columns=['order','concept:name'], inplace=True)
    long_test_df.drop(columns=['order','concept:name'], inplace=True)

    long_train_df.rename(columns={'prefix':'concept:name','trace_id':'Case ID'}, inplace=True)
    long_val_df.rename(columns={'prefix':'concept:name','trace_id':'Case ID'}, inplace=True)
    long_test_df.rename(columns={'prefix':'concept:name','trace_id':'Case ID'}, inplace=True)

    long_train_df['lifecycle:transition'] = 'complete'
    long_val_df['lifecycle:transition'] = 'complete'
    long_test_df['lifecycle:transition'] = 'complete'

    long_val_df = long_val_df[long_val_df['time:timestamp'] != 0]
    long_test_df = long_test_df[long_test_df['time:timestamp'] != 0]
    long_train_df = long_train_df[long_train_df['time:timestamp'] != 0]


    long_train_df['time:timestamp'] = pd.to_datetime(long_train_df['time:timestamp'], unit='s', errors='coerce')
    long_val_df['time:timestamp'] = pd.to_datetime(long_val_df['time:timestamp'], unit='s', errors='coerce')
    long_test_df['time:timestamp'] = pd.to_datetime(long_test_df['time:timestamp'], unit='s', errors='coerce')

    if not os.path.exists(CONF['data'].split('/')[0] + '/' + CONF['data'].split('/')[1] +'/' +  'imbalance_' + str(CONF['undersampling_factor'])):
        os.makedirs(CONF['data'].split('/')[0] + '/' + CONF['data'].split('/')[1] +'/' +  'imbalance_' + str(CONF['undersampling_factor']))
    long_train_df.to_csv(CONF['data'].split('/')[0] + '/' + CONF['data'].split('/')[1] +'/' +  'imbalance_' + str(CONF['undersampling_factor']) +'/'  + dataset_name + '_train' + '.csv', sep=',')
    long_val_df.to_csv(CONF['data'].split('/')[0] + '/' + CONF['data'].split('/')[1] +'/' +  'imbalance_' + str(CONF['undersampling_factor']) +'/'  + dataset_name + '_val' + '.csv', sep=',')
    long_test_df.to_csv(CONF['data'].split('/')[0] + '/' + CONF['data'].split('/')[1] +'/' +  'imbalance_' + str(CONF['undersampling_factor']) +'/'  + dataset_name + '_test' + '.csv', sep=',')
    '''
    test_df_convert = test_df.copy()
    encoder.decode(test_df_convert)
    # TEST DF TRANSFORM TO LOG
    cols = [*dataset_confs.dynamic_num_cols.values(), *dataset_confs.dynamic_cat_cols.values()]
    event_cols = list(itertools.chain.from_iterable(cols))+ ['prefix','time:timestamp']
    reconstructed_test_df = reconstruct_timestamps(test_df_convert)
    long_test_df = pd.wide_to_long(reconstructed_test_df, stubnames=event_cols, i='trace_id',j='order', sep='_', suffix=r'\w+').reset_index()
    long_test_df.drop(columns=['order'], inplace=True)
    long_test_df = long_test_df[long_test_df['time:timestamp'] != 0]
    long_test_df['time:timestamp'] = pd.to_datetime(long_test_df['time:timestamp'], unit='s', errors='coerce')
    long_test_df['start:timestamp'] = pd.to_datetime(long_test_df['start:timestamp'], unit='s', errors='coerce')
    long_test_df.rename(columns={'trace_id': 'Case ID','prefix':'Activity'}, inplace=True)
    test_log = long_test_df
    del long_test_df



    predictive_models = [PredictiveModel(CONF, predictive_model, train_df, val_df, test_df) for predictive_model in
                         CONF['predictive_models']]
    best_candidates, best_model_idx, best_model_model, best_model_config = retrieve_best_model(
        predictive_models,
        max_evaluations=CONF['hyperparameter_optimisation_evaluations'],
        target=CONF['hyperparameter_optimisation_target'],
        seed=CONF['seed']
    )
    best_model = predictive_models[best_model_idx]
    best_model.model = best_model_model
    best_model.config = best_model_config

    initial_feat_importance = np.argsort(best_model.model.feature_importances_)[::-1]
    initial_feat_importance = initial_feat_importance.astype('str')

    for index in range(len(initial_feat_importance)):
        initial_feat_importance[index] = train_df.columns[int(initial_feat_importance[index])]
    logger.debug('COMPUTE EXPLANATION')
    model_path = 'experiments/process_models/'
    support = 0.9

    encoder.decode(train_df)
    encoder.decode(val_df)

    reconstructed_train_val_log_df = pd.concat([train_df, val_df], ignore_index=True)

    encoder.encode(train_df)
    encoder.encode(val_df)
    if CONF['method'] == 'counterfactual':
        pred_df = reconstructed_train_val_log_df.copy()
        encoder.encode(pred_df)
        pred_df = pred_df[pred_df['label'] == y_maj]
        predicted_train = best_model.model.predict(drop_columns(pred_df))
        if best_model.model_type in [item.value for item in ClassificationMethods]:
            train_df_correct = pred_df[(pred_df['label'] == predicted_train)]
        else:
            train_df_correct = train_df_maj
        total_traces_to_gen = len(test_df)

        if CONF['feature_selection'] in ['simple', 'simple_trace']:
            cols = ['prefix']
        features_to_vary = None
        path_baseline_cfs = 'experiments/new_logs_icpm/' + dataset_name + '/results_cf/baseline_cf_df_' + dataset_name +'_' + str(CONF['undersampling_factor']) + '.csv'
        if os.path.exists(path_baseline_cfs):
            baseline_log = pd.read_csv(path_baseline_cfs)
            baseline_log['time:timestamp'] = pd.to_datetime(baseline_log['time:timestamp'], errors='coerce')
            baseline_log['start:timestamp'] = pd.to_datetime(baseline_log['start:timestamp'], errors='coerce')
            baseline_log['Case ID'] = baseline_log['Case ID'].astype('str')
            if dataset_name == 'BPI_Challenge_2012':
                baseline_log.drop(columns=['concept:name'], inplace=True)
                baseline_log.rename(
                    columns={'label': 'case:label', 'AMOUNT_REQ': 'case:AMOUNT_REQ', 'Case ID': 'case:concept:name',
                             'Activity': 'concept:name'}, inplace=True)
            elif dataset_name == 'sepsis':
                cols = [*dataset_confs.static_num_cols.values(), *dataset_confs.static_cat_cols.values(), ['label']]
                cols = list(itertools.chain.from_iterable(cols))
                cols = cols
                cols_for_traces = ['case:' + col for col in cols]

                baseline_log.rename(columns=dict(zip(cols, cols_for_traces)), inplace=True)
                baseline_log.rename(
                    columns={'Case ID': 'case:concept:name',
                             'Activity': 'concept:name'}, inplace=True)

            baseline_log = pm4py.convert_to_event_log(baseline_log, timestamp_key='time:timestamp',
                                                      case_id_key='case:concept:name', activity_key='concept:name',
                                                      resource_key='Resource')
            _, baseline_df = get_encoded_df(log=baseline_log, CONF=CONF, encoder=encoder)
            encoder.decode(baseline_df)
            df_cf = baseline_df.copy()
        else:
            os.makedirs('experiments/new_logs_icpm/' + dataset_name + '/results_cf', exist_ok=True)
            df_cf, x_eval = explain(CONF, best_model, encoder=encoder,
                            query_instances=train_df_correct,
                            method='genetic', df=full_df.iloc[:, 1:], optimization='baseline',
                            heuristic='heuristic_2', support=support,
                            timestamp_col_name=[*dataset_confs.timestamp_col.values()][0],
                            model_path=model_path, random_seed=CONF['seed'], total_traces=total_traces_to_gen,
                            minority_class=y_min, cfs_to_gen=1 #how many cfs to generate at one time
                                    , features_to_vary=features_to_vary
                            )
            df_cf.rename(columns={'Case ID':'trace_id'},inplace=True)

            reconstructed_df_cf = reconstruct_timestamps(df_cf)
            long_df_cf = pd.wide_to_long(reconstructed_df_cf, stubnames=event_cols, i='trace_id', j='order',
                                           sep='_', suffix=r'\w+').reset_index()
            long_df_cf.drop(columns=['order'], inplace=True)
            long_df_cf = long_df_cf[long_df_cf['time:timestamp'] != 0]
            long_df_cf['time:timestamp'] = pd.to_datetime(long_df_cf['time:timestamp'], unit='s', errors='coerce')
            long_df_cf['start:timestamp'] = pd.to_datetime(long_df_cf['start:timestamp'], unit='s', errors='coerce')
            long_df_cf.rename(columns={'trace_id': 'Case ID', 'prefix': 'Activity'}, inplace=True)
            baseline_log = long_df_cf
            baseline_log.to_csv(path_baseline_cfs, index=False)

        reconstructed_df_cf = df_cf.copy()

        path_full_simulated = 'experiments/new_logs_icpm/' + dataset_name + '/results_cf/simulated_log_full_' + dataset_name +'_' + str(CONF['undersampling_factor']) + '.csv'
        path_simulated_cfs = 'experiments/new_logs_icpm/' + dataset_name + '/results_cf/simulated_log_test_only_' + dataset_name +'_' + str(CONF['undersampling_factor']) + '.csv'


        ### simulation part
        if os.path.exists(path_simulated_cfs):
            print('Simulated log already exists')
        else:
            if CONF['simulation']:
                run_simulation(reconstructed_train_val_log_df, reconstructed_df_cf, dataset_name, CONF['undersampling_factor'],path_full_simulated)
                simulated_log = pd.read_csv(path_full_simulated)

                simulated_log.rename(columns={'id_case': 'trace_id'}, inplace=True)
                simulated_log = simulated_log[simulated_log['trace_id'].str.contains('_CF')].reset_index()

                simulated_log.to_csv(path_simulated_cfs)
        if CONF['simulation']:
            simulated_log = pd.read_csv(path_simulated_cfs)
            if 'Unnamed: 0' in simulated_log.columns:
                simulated_log.drop(columns=['Unnamed: 0'], inplace=True)
            simulated_log.rename(columns={'trace_id': 'Case ID', 'activity': 'Activity', 'start_time': 'start_timestamp',
                                          'end_time': 'time_timestamp', 'resource': 'Resource'}, inplace=True)
            simulated_log.drop(columns=['role', 'prefix', 'enabled_time','index'], inplace=True)

        #x_eval.to_csv('experiments/new_logs_icpm/' + dataset_name + '/results/cf_eval' + dataset_name +'_' + str(CONF['undersampling_factor']) + '.csv')
    elif CONF['method'] == 'kappel':
        # TODO, implement kappel method
        pass

    elif CONF['method'] == 'cvae':
        baseline_vae_log = 'experiments/new_logs_icpm/' + dataset_name + '/results_vae/gen_' + str(CONF['undersampling_factor']) + '.xes'
        path_simulated_log = 'experiments/new_logs_icpm/' + dataset_name + '/results_vae/gen_sim' + str(
            CONF['undersampling_factor']) + '.csv'
        path_sim_gen_test = 'experiments/new_logs_icpm/' + dataset_name + '/results_vae/gen_sim_test_only' + dataset_name + '_' + str(
            CONF['undersampling_factor']) + '.csv'
        if os.path.exists(baseline_vae_log):

            print('Baseline log already exists')
            baseline_log = pm4py.read_xes(baseline_vae_log)
            baseline_log['time:timestamp'] = pd.to_datetime(
                baseline_log['time:timestamp'], errors='coerce'
            ).apply(lambda x: x.tz_localize(None) if pd.notnull(x) and x.tzinfo is not None else x)
            baseline_log['start:timestamp'] = baseline_log['time:timestamp']
            baseline_log['start:timestamp'] = pd.to_datetime(
                baseline_log['start:timestamp'], errors='coerce'
            ).apply(lambda x: x.tz_localize(None) if pd.notnull(x) and x.tzinfo is not None else x)
            baseline_log.drop(columns=['relative_timestamp_from_start',
                                       'relative_timestamp_from_previous_activity', 'Unnamed: 0','Case ID','Activity','org:resource'],inplace=True)
            cols = [*dataset_confs.static_num_cols.values(), *dataset_confs.static_cat_cols.values(),['label']]
            cols = list(itertools.chain.from_iterable(cols))
            cols = cols
            cols_for_traces = ['case:' + col for col in cols]

            baseline_log.rename(columns=dict(zip(cols, cols_for_traces)), inplace=True)
            baseline_log = baseline_log.iloc[:,:-1]
            baseline_log = pm4py.convert_to_event_log(baseline_log, timestamp_key='time:timestamp',
                                                       case_id_key='case:concept:name', activity_key='concept:name',
                                                       resource_key='Resource')
        else:
            print('Baseline log does not exist')
        if os.path.exists(path_simulated_log):
            print('Simulated log already exists')
            simulated_log = pd.read_csv(
                path_sim_gen_test).reset_index(drop=True)

            baseline_log = pm4py.convert_to_dataframe(baseline_log, timestamp_key='time:timestamp',
                                                      case_id_key='case:concept:name', activity_key='concept:name')
        else:
            _, baseline_df = get_encoded_df(log=baseline_log, CONF=CONF, encoder=encoder)

            encoder.decode(baseline_df)

            run_simulation(reconstructed_train_val_log_df, baseline_df, dataset_name, CONF['undersampling_factor'],
                           path_result=path_simulated_log)


            simulated_log = pd.read_csv(path_simulated_log)
            simulated_log = simulated_log[simulated_log['id_case'].str.contains('_CF')].reset_index()

            simulated_log.to_csv(path_sim_gen_test)
            baseline_log = pm4py.convert_to_dataframe(baseline_log, timestamp_key='time:timestamp',
                                                      case_id_key='case:concept:name', activity_key='concept:name')


        simulated_log.rename(columns={'id_case': 'Case ID', 'activity': 'Activity', 'start_time': 'start_timestamp',
                                      'end_time': 'time_timestamp', 'resource': 'Resource'}, inplace=True)
        if 'role' in simulated_log.columns:
            simulated_log.drop(columns=['role', 'prefix', 'enabled_time'], inplace=True)

        if 'Unnamed: 0' in simulated_log.columns:
            simulated_log.drop(columns=['Unnamed: 0'], inplace=True)
        if 'index' in simulated_log.columns:
            simulated_log.drop(columns=['index'], inplace=True)

    evaluations = []
    if CONF['simulation']:
    # First evaluation: simulated log
        gen_eval_sim = logs_evaluation(
            original_log=test_log,
            generated_log=simulated_log,
            timestamp_col_name='time:timestamp',
            CONF=CONF,
            dataset=dataset_name,
            model_path=model_path
        )
        gen_eval_sim['method'] = CONF['method']
        gen_eval_sim['prefix_length'] = CONF['prefix_length']
        gen_eval_sim['undersampling_factor'] = CONF['undersampling_factor']
        gen_eval_sim['dataset'] = dataset_name
        gen_eval_sim['label_to_gen'] = CONF['label_to_gen']
        gen_eval_sim['simulation'] = CONF['simulation']
        evaluations.append(gen_eval_sim)
    if CONF['method'] == 'cvae':
    # Second evaluation: baseline log
        baseline_log.rename(columns=dict(zip(cols_for_traces, cols)), inplace=True)
        try:
            baseline_log.rename(columns={'case:concept:name': 'Case ID', 'concept:name': 'Activity'}, inplace=True)
        except:
            print('No columns to rename')

    baseline_log = pm4py.convert_to_dataframe(baseline_log)
    baseline_log.rename(columns={'case:concept:name': 'Case ID', 'concept:name': 'Activity'}, inplace=True)
    gen_eval_baseline = logs_evaluation(
        original_log=test_log,
        generated_log=baseline_log,
        timestamp_col_name='time:timestamp',
        CONF=CONF,
        dataset=dataset_name,
        model_path=model_path
    )
    gen_eval_baseline['method'] = CONF['method']
    gen_eval_baseline['prefix_length'] = CONF['prefix_length']
    gen_eval_baseline['undersampling_factor'] = CONF['undersampling_factor']
    gen_eval_baseline['dataset'] = dataset_name
    gen_eval_baseline['label_to_gen'] = CONF['label_to_gen']
    gen_eval_baseline['simulation'] = 'False'
    evaluations.append(gen_eval_baseline)

    # Convert to DataFrame
    generation_evaluation_df = pd.DataFrame(evaluations)

    # Save to CSV
    if not os.path.exists('experiments/new_logs_icpm/' + dataset_name + '/results_evaluation'):
        os.makedirs('experiments/new_logs_icpm/' + dataset_name + '/results_evaluation')
    output_path = f'experiments/new_logs_icpm/{dataset_name}/results_evaluation/generation_evaluation_updated_sim_{dataset_name}.csv'
    generation_evaluation_df.to_csv(output_path, index=False, mode='a', header=not os.path.exists(output_path))

    logger.info('RESULT')

    logger.info('RESULT')
    logger.info('Done, cheers!')


if __name__ == '__main__':
    dataset_list = {
        ### prefix length
        'BPI_Challenge_2012': [45],
        #'sepsis_cases_2_start': [12],
        #'bpic2015_2_start': [55],
        #'bpic2015_2_start': [12],
        #'bpic2015_2_start': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ,14 ,15],
        #'BPI17': [20],
        #'ConsultaDataMining201618': [9]
        #'Productions': [40]
        #'PurchasingExample': [40]
        #"cvs_pharmacy": [8]
        #'sepsis': [25]
    }
    #factors = [0.3, 0.2, 0.15, 0.1, 0.05]
    factors = [0.2]
    for dataset, prefix_lengths in dataset_list.items():
         print(os.path.join('datasets', dataset, 'full_label.xes'))
         for factor in factors:
            for prefix in prefix_lengths:
                CONF = {  # This contains the configuration for the run
                    'data': os.path.join('datasets',dataset, 'full_label.xes'),
                    'train_val_test_split': [0.8, 0.15, 0.05],
                    'output': os.path.join('..', 'output_data'),
                    'prefix_length_strategy': PrefixLengthStrategy.FIXED.value,
                    'prefix_length': prefix,
                    'padding': True,  # TODO, why use of padding?
                    'feature_selection': EncodingType.COMPLEX.value,
                    'task_generation_type': TaskGenerationType.ONLY_THIS.value,
                    'attribute_encoding': EncodingTypeAttribute.LABEL.value,  # LABEL, ONEHOT
                    'labeling_type': LabelTypes.ATTRIBUTE_STRING.value,
                    'predictive_models': [ClassificationMethods.XGBOOST.value],  # RANDOM_FOREST, LSTM, PERCEPTRON
                    'explanator': ExplainerType.DICE_AUGMENTATION.value,
                    'threshold': 13,
                    'top_k': 10,
                    'hyperparameter_optimisation': False,  # TODO, this parameter is not used
                    'hyperparameter_optimisation_target': HyperoptTarget.AUC.value,
                    'hyperparameter_optimisation_evaluations': 20,
                    'time_encoding': TimeEncodingType.NONE.value,
                    'target_event': None,
                    'seed': 666,
                    'simulation': True,  # if True the simulation of TRAIN + CF is run,
                    'drop_factuals': False,
                    'label_to_gen': 'deviant',# regular or deviant
                    'undersampling_factor':factor, # how much to retain from the undersampled class for training
                    'method':'counterfactual' #method for data generation: kappel, cvae, counterfactual
                }
                run_simple_pipeline(CONF=CONF, dataset_name=dataset)
