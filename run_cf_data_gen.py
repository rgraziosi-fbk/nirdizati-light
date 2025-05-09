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
import random
import json
from pm4py import convert_to_event_log, write_xes
from dataset_confs import DatasetConfs
from new_rims.run_simulation import run_simulation
import ast

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


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
        reconstructed_df = df.copy()  # Avoid modifying the original DataFrame

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
    train_df_min = full_df_min.sample(frac=CONF['undersampling_factor'], random_state=42)
    #Remove the 20%
    test_df = full_df_min.drop(train_df_min.index)

    train_df = pd.concat([train_df_maj, train_df_min])
    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']

    from sklearn.model_selection import StratifiedShuffleSplit,ShuffleSplit
    ss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    for train_index, val_index in ss_val_test.split(X_train, y_train):
        X_train, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

    # Now you have your splits
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
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
    if CONF['explanator'] is ExplainerType.DICE_AUGMENTATION.value:
        predicted_test = best_model.model.predict(drop_columns(test_df))
        predicted_train = best_model.model.predict(drop_columns(train_df))
        if best_model.model_type in [item.value for item in ClassificationMethods]:
            train_df_correct = train_df[(train_df['label'] == predicted_train)]
        else:
            train_df_correct = train_df_correct
        train_df_correct = train_df_correct[train_df_correct['label'] == y_maj]
        total_traces_to_gen = len(test_df)


        model_path = 'experiments/process_models/'
        support = 0.9
        import itertools
        if CONF['feature_selection'] in ['simple', 'simple_trace']:
            cols = ['prefix']
        features_to_vary = None

        df_cf, x_eval = explain(CONF, best_model, encoder=encoder,
                        query_instances=train_df_correct,
                        method='genetic', df=full_df.iloc[:, 1:], optimization='baseline',
                        heuristic='heuristic_2', support=support,
                        timestamp_col_name=[*dataset_confs.timestamp_col.values()][0],
                        model_path=model_path, random_seed=CONF['seed'], total_traces=total_traces_to_gen,
                        minority_class=y_min, cfs_to_gen=1 #how many cfs to generate at one time
                                , features_to_vary=features_to_vary
                        )
        if CONF['drop_factuals']:
            train_df = train_df[~train_df.trace_id.isin(df_cf['Case ID'])]
        df_cf.rename(columns={'Case ID':'trace_id'},inplace=True)
        encoder.decode(train_df)
        encoder.decode(val_df)

        reconstructed_train_val_log_df = pd.concat([train_df, val_df], ignore_index=True)
        reconstructed_df_cf = df_cf.copy()

        ### simulation part
        if CONF['simulation']:
            run_simulation(reconstructed_train_val_log_df, reconstructed_df_cf, dataset_name)
            path_simulated_cfs = os.getcwd()+'/experiments/icpm_data_gen_eval/datasets/' + dataset_name + '/results/simulated_log_' + dataset_name + '_.csv'
            simulated_log = pd.read_csv(path_simulated_cfs)
            dicts_trace = {}
            for i in range(len(simulated_log)):
                dicts_trace[i] = ast.literal_eval(simulated_log.loc[i][-2])
            df = pd.DataFrame.from_dict(dicts_trace, orient='index')
            try:
                simulated_log = pd.merge(simulated_log, df, how='inner', on=df.index)
            except Exception as e:
                print(e)
            try:
                simulated_log.drop(columns=['key_0','st_tsk_wip', 'queue', 'arrive:timestamp', 'attrib_trace'], inplace=True)
            except Exception as e:
                simulated_log.drop(columns=['st_tsk_wip', 'queue', 'arrive:timestamp', 'attrib_trace'], inplace=True)
            simulated_log.rename(columns={'queue.1': 'queue'}, inplace=True)
            if dataset_name == 'cvs_pharmacy' or dataset_name == 'ConsultaDataMining201618' or dataset_name == 'SynLoan' or dataset_name == 'PurchasingExample' or dataset_name == 'Productions' or dataset_name == 'BPI_Challenge_2012_W_Two_TS' or dataset_name == 'bpic2015_4_start' or dataset_name == 'sepsis_cases_2_start':
                simulated_log.drop(columns=['open_cases'], inplace=True)
            simulated_log.rename(
                    columns={'role': 'org:resource', 'task': 'concept:name', 'caseid': 'case:concept:name'}, inplace=True)
            if dataset_name == 'sepsis_cases_1_start' or dataset_name == 'sepsis_cases_2_start' or dataset_name == 'sepsis_cases_3_start':
                simulated_log['org:group'] = simulated_log['org:resource']
            simulated_log['lifecycle:transition'] = 'complete'
            cols = [*dataset_confs.static_cat_cols.values(), *dataset_confs.static_num_cols.values()]
            cols = list(itertools.chain.from_iterable(cols))
            for i in range(len(simulated_log)):
                for x in cols:
                    simulated_log.at[i, x] = dicts_trace[i][x]
            cols.append('label')
            simulated_log['time:timestamp'] = pd.to_datetime(simulated_log['time:timestamp'], utc=True)
            simulated_log['start:timestamp'] = pd.to_datetime(simulated_log['start:timestamp'], utc=True)
            if dataset_name != 'SynLoan':
                simulated_log.drop(columns=[col for col in simulated_log.columns if 'transition' in col], inplace=True)
            #simulated_log['label'] = minority_class
            simulated_log = convert_to_log(simulated_log, cols)
            _, simulated_df = get_encoded_df(log=simulated_log, encoder=encoder, CONF=CONF)
            updated_test_df = simulated_df.copy()
            encoder.decode(updated_test_df)
            encoder.decode(simulated_df)
            updated_test_df = reconstruct_timestamps(updated_test_df)
            #simulated_df.to_csv(os.path.join('experiments', dataset_name + '_train_sim.csv'))
            updated_test_df.to_csv(path_or_buf=os.path.join('experiments', 'new_logs_icpm', dataset_name,
                                                             dataset_name + '_test_df_cf_simulated_aug_' + str(
                                                                 augmentation_factor) + '_pref_len_' + str(
                                                                 CONF['prefix_length']) + '.csv'), index=False)
        else:
            updated_test_df = pd.concat([reconstructed_test_df, reconstructed_test_df], ignore_index=True)
            updated_test_df.to_csv(path_or_buf=os.path.join('experiments', 'new_logs_icpm', dataset_name,
                                                             dataset_name + '_test_df_cf_aug_' + str(
                                                                 augmentation_factor) + '_pref_len_' + str(
                                                                 CONF['prefix_length']) + '.csv'), index=False)
            x_eval.to_csv(path_or_buf=os.path.join('experiments', 'cf_eval_results', dataset_name + '_cf_eval' + str(
                augmentation_factor) + '_pref_len_' + str(CONF['prefix_length']) + '.csv'), index=False)
            #updated_train_df.to_csv(os.path.join('experiments', dataset_name + '_train_baseline.csv'))
            updated_test_df.to_csv(os.path.join('experiments', dataset_name + '_train_baseline.csv'))
            #encoder.encode(updated_train_df)
        # Have to do the prefixes loop here to get the results for each prefix length, train each predictive model again, add the counterfactuals and retrain with the updated_train_df
        #updated_train_df = pd.read_csv(os.path.join('experiments', dataset_name + '_train_sim.csv'),index_col=[0])

    logger.info('RESULT')
    logger.info('Done, cheers!')


if __name__ == '__main__':
    dataset_list = {
        ### prefix length
        #'bpic2012_2_start_old': [45],
        #'sepsis_cases_2_start': [12],
        #'bpic2015_2_start': [55],
        #'bpic2015_2_start': [12],
        #'bpic2015_2_start': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ,14 ,15],
        #'BPI17': [20],
        #'ConsultaDataMining201618': [9]
        #'Productions': [40]
        #'PurchasingExample': [40]
        #"cvs_pharmacy": [8]
        'sepsis': [25],
    }
    factors = [0.2, 0.1, 0.05, 0.01]
    for dataset, prefix_lengths in dataset_list.items():
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
                    'simulation': True,  ## if True the simulation of TRAIN + CF is run,
                    'drop_factuals': False,
                    'label_to_gen': 'deviant',# regular or deviant
                    'undersampling_factor':factor#how much to retain from the undersampled class for training
                }
                run_simple_pipeline(CONF=CONF, dataset_name=dataset)
