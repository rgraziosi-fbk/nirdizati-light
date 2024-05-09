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
from run_simulation import run_simulation
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

    logger.debug('TRAIN PREDICTIVE MODEL')
    # split in train, val, test
    train_size = CONF['train_val_test_split'][0]
    val_size = CONF['train_val_test_split'][1]
    test_size = CONF['train_val_test_split'][2]
    if train_size + val_size + test_size != 1.0:
        raise Exception('Train-val-test split doese  not sum up to 1')

    full_df = full_df[full_df.columns.drop(list(full_df.filter(regex='Resource')))]
    full_df = full_df[full_df.columns.drop(list(full_df.filter(regex='Activity')))]
    train_df, val_df, test_df = np.split(full_df, [int(train_size*len(full_df)), int((train_size+val_size)*len(full_df))])

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
    logger.debug('COMPUTE EXPLANATION')
    if CONF['explanator'] is ExplainerType.DICE_AUGMENTATION.value:
        # set test df just with correctly predicted labels and make sure it's minority class
        minority_class = full_df['label'].value_counts().idxmin()
        majority_class = full_df['label'].value_counts().idxmax()
        predicted_train = best_model.model.predict(drop_columns(train_df))
        if best_model.model_type in [item.value for item in ClassificationMethods]:
            train_df_correct = train_df[(train_df['label'] == predicted_train) & (train_df['label'] == majority_class)]
        else:
            train_df_correct = train_df
        full_df = pd.concat([train_df, val_df, test_df])
        augmentation_factor = CONF['augmentation_factor']
        total_traces = augmentation_factor * len(train_df[train_df['label'] == majority_class])
        model_path = 'experiments/process_models/'
        support = 0.9
        import itertools
        if CONF['feature_selection'] in ['simple', 'simple_trace']:
            cols = ['prefix']
        elif CONF['feature_selection']:
            cols = [*dataset_confs.dynamic_num_cols.values(), *dataset_confs.dynamic_cat_cols.values()]
            cols = list(itertools.chain.from_iterable(cols))
            cols.append('time:timestamp')
            cols.append('prefix')
            cols.append('lifecycle:transition')
        df_cf,x_eval = explain(CONF, best_model, encoder=encoder,
                        query_instances=train_df_correct,
                        method='genetic', df=full_df.iloc[:, 1:], optimization='baseline',
                        heuristic='heuristic_2', support=support,
                        timestamp_col_name=[*dataset_confs.timestamp_col.values()][0],
                        model_path=model_path, random_seed=CONF['seed'], total_traces=total_traces,
                        minority_class=minority_class,cfs_to_gen=10#how many cfs to generate at one time
                        )
        df_cf.drop(columns=['Case ID'], inplace=True)
        encoder.decode(train_df)

        train_df.to_csv(os.path.join('experiments',dataset_name + '_train_df.csv'))
        df_cf['trace_id'] = df_cf.index
        df_cf.to_csv(os.path.join('experiments', dataset_name + '_cf.csv'), index=False)


        ### simulation part
        if CONF['simulation']:
            run_simulation(train_df, df_cf, dataset_name)
            path_simulated_cfs = dataset_name + '/results/simulated_log_' + dataset_name + '_.csv'
            simulated_log = pd.read_csv(path_simulated_cfs)
            dicts_trace = {}
            for i in range(len(simulated_log)):
                dicts_trace[i] = ast.literal_eval(simulated_log.loc[i][-2])
            df = pd.DataFrame.from_dict(dicts_trace, orient='index')
            simulated_log = pd.merge(simulated_log, df, how='inner', on=df.index)
            simulated_log.drop(columns=['key_0',
                                        'st_tsk_wip', 'queue', 'arrive:timestamp', 'attrib_trace'], inplace=True)
            if dataset_name == 'BPI_Challenge_2012_W_Two_TS' or dataset_name == 'bpic2015_2_start':
                simulated_log.drop(columns=['open_cases'], inplace=True)
            simulated_log.rename(
                columns={'role': 'org:resource', 'task': 'concept:name', 'caseid': 'case:concept:name'}, inplace=True)
            if dataset_name == 'sepsis_cases_1_start':
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
            #simulated_log['label'] = minority_class
            simulated_log = convert_to_log(simulated_log, cols)
            _, simulated_df = get_encoded_df(log=simulated_log, encoder=encoder, CONF=CONF)
            simulated_df.to_csv(os.path.join('experiments', dataset_name + '_train_sim.csv'))
            updated_train_df = simulated_df.copy()
        else:
            updated_train_df = pd.concat([train_df, df_cf], ignore_index=True)
            updated_train_df.to_csv(path_or_buf=os.path.join('experiments', 'new_logs',
                                                             dataset_name + '_train_df_cf_aug_' + str(
                                                                 augmentation_factor) + '_pref_len_' + str(
                                                                 CONF['prefix_length']) + '.csv'), index=False)
            x_eval.to_csv(path_or_buf=os.path.join('experiments', 'cf_eval_results', dataset_name + '_cf_eval' + str(
                augmentation_factor) + '_pref_len_' + str(CONF['prefix_length']) + '.csv'), index=False)
            updated_train_df.to_csv(os.path.join('experiments', dataset_name + '_train_baseline.csv'))
            encoder.encode(updated_train_df)

        #updated_train_df = pd.read_csv(os.path.join('experiments', dataset_name + '_train_sim.csv'),index_col=[0])
        predictive_models_new = [PredictiveModel(CONF, predictive_model, updated_train_df, val_df, test_df) for predictive_model in
                             CONF['predictive_models']]
        best_candidates_new, best_model_idx_new, best_model_model_new, best_model_config_new = retrieve_best_model(
            predictive_models_new,
            max_evaluations=CONF['hyperparameter_optimisation_evaluations'],
            target=CONF['hyperparameter_optimisation_target'],
            seed=CONF['seed']
        )

        best_model_new = predictive_models_new[best_model_idx_new]
        best_model.model = best_model_model_new
        best_model.config = best_model_config_new

        data_list = []

        # Iterate over the best candidates
        for id, _ in enumerate(best_candidates):
            initial_result = best_candidates[id].get('result')
            augmented_result = best_candidates_new[id].get('result')
            model = CONF['predictive_models'][id]
            prefix_length = CONF['prefix_length']
            augmentation_factor = augmentation_factor
            simulation = CONF['simulation']

            columns = []
            for key in initial_result.keys:
                columns.append(f'Initial {key.capitalize()}')
            for key in augmented_result.keys:
                columns.append(f'Augmented {key.capitalize()}')
            columns.extend(['Model', 'Prefix Length', 'Augmentation Factor', 'Simulation'])

            # Create an empty DataFrame
            results_df = pd.DataFrame(columns=columns)

            # Create a data row
            data_row = []

            # Append initial metrics
            for key in initial_result.keys:
                data_row.append(initial_result[key])

            # Append augmented metrics
            for key in augmented_result.keys:
                data_row.append(augmented_result[key])

            # Append additional data to the data row
            data_row.extend([model, prefix_length, augmentation_factor, simulation])

            # Append the data row to the DataFrame
            results_df = results_df.append(pd.Series(data_row, index=columns), ignore_index=True)

        # Define the file path
        file_path = 'experiments/model_performances_' + CONF['hyperparameter_optimisation_target'] + '_' + dataset_name+ '.csv'

        # Write the DataFrame to a CSV file in append mode
        results_df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

        print("New results appended to the CSV file.")

    logger.info('RESULT')
    logger.info('Done, cheers!')


if __name__ == '__main__':
    dataset_list = {
        ### prefix length
        #'BPI_Challenge_2012_W_Two_TS': [3],
        'bpic2015_2_start': [3, 5],
        #'sepsis_cases_1_start': [5],
    }
    for dataset, prefix_lengths in dataset_list.items():
        for prefix in prefix_lengths:
            for augmentation_factor in [0.3, 0.5, 0.7]:
                CONF = {  # This contains the configuration for the run
                    'data': os.path.join(dataset, 'bpic2015_2_start.xes'),
                    'train_val_test_split': [0.7, 0.15, 0.15],
                    'output': os.path.join('..', 'output_data'),
                    'prefix_length_strategy': PrefixLengthStrategy.FIXED.value,
                    'prefix_length': prefix,
                    'padding': True,  # TODO, why use of padding?
                    'feature_selection': EncodingType.COMPLEX.value,
                    'task_generation_type': TaskGenerationType.ONLY_THIS.value,
                    'attribute_encoding': EncodingTypeAttribute.LABEL.value,  # LABEL, ONEHOT
                    'labeling_type': LabelTypes.ATTRIBUTE_STRING.value,
                    'predictive_models': [ClassificationMethods.XGBOOST.value, ClassificationMethods.RANDOM_FOREST.value],  # RANDOM_FOREST, LSTM, PERCEPTRON
                    'explanator': ExplainerType.DICE_AUGMENTATION.value,
                    'augmentation_factor': augmentation_factor,# SHAP, LRP, ICE, DICE
                    'threshold': 13,
                    'top_k': 10,
                    'hyperparameter_optimisation': False,  # TODO, this parameter is not used
                    'hyperparameter_optimisation_target': HyperoptTarget.MCC.value,
                    'hyperparameter_optimisation_evaluations': 20,
                    'time_encoding': TimeEncodingType.NONE.value,
                    'target_event': None,
                    'seed': 42,
                    'simulation': False  ## if True the simulation of TRAIN + CF is run
                }
                run_simple_pipeline(CONF=CONF, dataset_name=dataset)
