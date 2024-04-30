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

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


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
        raise Exception('Train-val-test split does not sum up to 1')

    full_df = full_df[full_df.columns.drop(list(full_df.filter(regex='Resource')))]
    full_df = full_df[full_df.columns.drop(list(full_df.filter(regex='Activity')))]
    train_df,val_df,test_df = np.split(full_df,[int(train_size*len(full_df)), int((train_size+val_size)*len(full_df))])

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
        predicted= best_model.model.predict(drop_columns(test_df))
        if best_model.model_type in [item.value for item in ClassificationMethods]:
            train_df_correct = train_df[(train_df['label'] == predicted_train) & (train_df['label'] == majority_class)]
            test_df_correct = test_df[(test_df['label'] == predicted) & (test_df['label'] == majority_class)]
        else:
            train_df_correct = train_df
            test_df_correct = test_df
        cf_dataset = pd.concat([train_df, val_df], ignore_index=True)
        full_df = pd.concat([train_df, val_df, test_df])
        cf_dataset.loc[len(cf_dataset)] = 0
        augmentation_factor = CONF['augmentation_factor']
        #methods = ['genetic_conformance']
        #optimizations = ['filtering','loss_function']
        ## TOTAL TRACES = HOW MANY TO GENERATE, ADD AUGMENTATION FACTOR INSTEAD
        total_traces = augmentation_factor * len(full_df[full_df['label']==majority_class])
        method = 'genetic'
        optimization = 'baseline'
        heuristic = 'heuristic_2'
        model_path = '../experiments/process_models/process_models'
        support = 1.0
        import itertools
        if CONF['feature_selection'] in ['simple','simple_trace']:
            cols = ['prefix']
        elif CONF['feature_selection']:
            cols = [*dataset_confs.dynamic_num_cols.values(), *dataset_confs.dynamic_cat_cols.values()]
            cols = list(itertools.chain.from_iterable(cols))
            cols.append('time:timestamp')
            cols.append('prefix')
            cols.append('lifecycle:transition')
        simulated_cfs = None
        if simulated_cfs:
            encoder.decode(train_df)

            long_data = pd.wide_to_long(train_df, stubnames=cols, i='trace_id',
                                        j='order', sep='_', suffix=r'\w+')
            long_data_sorted = long_data.sort_values(['trace_id', 'order'], ).reset_index(drop=False)
            long_data_sorted.drop(columns=['order'], inplace=True)
            columns_to_rename = {'trace_id': 'case:concept:name'}
            columns_to_rename.update({'prefix': 'concept:name'})
            long_data_sorted.rename(columns=columns_to_rename, inplace=True)
            long_data_sorted['label'].replace({'regular': 'false', 'deviant': 'true'}, inplace=True)
            long_data_sorted.replace('0', 'other', inplace=True)
            train_log = convert_to_event_log(long_data_sorted)
            import ast
            if 'sepsis' in simulated_cfs:
                if simulated_cfs.endswith('.xes'):
                    simulated_log = get_log(filepath=simulated_cfs)
                    simulated_log = pm4py.convert_to_dataframe(simulated_log)
                    dicts = {}
                    for i in range(len(simulated_log)):
                        dicts[i] = ast.literal_eval(simulated_log.loc[i][-1])
                    df = pd.DataFrame.from_dict(dicts, orient='index')
                    simulated_log = pd.merge(simulated_log, df, how='inner', on=df.index)
                    simulated_log.drop(columns=['key_0', 'amount', 'st_wip',
                                                'st_tsk_wip', 'queue', 'arrive:timestamp'], inplace=True)
                    simulated_log.rename(
                        columns={'role': 'org:resource', 'task': 'concept:name', 'caseid': 'case:concept:name'},
                        inplace=True)
                    simulated_log['org:group'] = simulated_log['org:resource']
                    simulated_log['label'] = 'true'
                    simulated_log['lifecycle:transition'] = 'complete'
                else:
                    simulated_log = pd.read_csv(simulated_cfs)
                    dicts = {}
                    for i in range(len(simulated_log)):
                        dicts[i] = ast.literal_eval(simulated_log.loc[i][-1])
                    df = pd.DataFrame.from_dict(dicts, orient='index')
                    simulated_log = pd.merge(simulated_log, df, how='inner', on=df.index)
                    simulated_log.drop(columns=['key_0', 'amount', 'st_wip',
                     'st_tsk_wip','queue','arrive:timestamp'],inplace=True)
                    simulated_log.rename(columns={'role': 'org:resource','task':'concept:name','caseid':'case:concept:name'}, inplace=True)
                    simulated_log['org:group'] = simulated_log['org:resource']
                    simulated_log['label'] = 'true'
                    simulated_log['lifecycle:transition'] = 'complete'

            train_log = pm4py.convert_to_dataframe(train_log)
            if '2012' in simulated_log:
                simulated_log = simulated_log.rename(columns={'amount': 'AMOUNT_REQ', 'role': 'org:resource'})
                simulated_log['lifecycle:transition'] = 'complete'
                simulated_log['label'] = 'true'

            simulated_log = simulated_log[train_log.columns]
            simulated_log = pm4py.convert_to_event_log(simulated_log)

            if '2012' in dataset_name:
                for i in range(len(simulated_log)):
                    for x in range(len(cols)):
                        simulated_log[i].attributes.update({cols[x]: simulated_log[i][0]._dict[cols[x]]})
                    for j in range(len(simulated_log[i])):
                        for x in range(len(cols)):
                            del simulated_log[i][j]._dict[cols[x]]
            elif 'sepsis' in dataset_name:
                cols = [ *dataset_confs.static_cat_cols.values(), *dataset_confs.static_num_cols.values()]
                cols = list(itertools.chain.from_iterable(cols))
                cols.append('label')
                for i in range(len(simulated_log)):
                    for x in range(len(cols)):
                        simulated_log[i].attributes.update({cols[x]: simulated_log[i][0]._dict[cols[x]]})
                    for j in range(len(simulated_log[i])):
                        for x in range(len(cols)):
                            del simulated_log[i][j]._dict[cols[x]]
            encoder, simulated_df = get_encoded_df(log=simulated_log,encoder=encoder,CONF=CONF)
            predicted_simulated = predictive_model.model.predict(drop_columns(simulated_df))
            simulated_df['label'] = predicted_simulated
            updated_train_df = simulated_df
        else:
            df_cf = explain(CONF, best_model, encoder=encoder,
                            query_instances=train_df_correct,
                            method=method, df=full_df.iloc[:, 1:], optimization=optimization,
                            heuristic=heuristic, support=support,
                            timestamp_col_name=[*dataset_confs.timestamp_col.values()][0],
                            model_path=model_path, random_seed=CONF['seed'], total_traces=total_traces,
                            minority_class=minority_class
                            # ,case_ids=case_ids
                            )
            case_ids = df_cf['Case ID']
            df_cf.drop(columns=['Case ID'], inplace=True)
            encoder.encode(df_cf)
            updated_df = pd.concat([train_df,df_cf,test_df,val_df], ignore_index=True)
            updated_df['trace_id'] = updated_df.index
            encoder.decode(updated_df)

            encoder.decode(train_df)
            encoder.decode(df_cf)


            train_df.to_csv(os.path.join('..','experiments',dataset_name + 'train_df.csv'))
            df_cf['trace_id'] = df_cf.index
            df_cf.to_csv(os.path.join('..','experiments',dataset_name + '_cf.csv'), index=False)

            updated_train_df = pd.concat([train_df,df_cf], ignore_index=True)
            encoder.encode(updated_train_df)

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

        #print('Original model',initial_result,'\n','Updated model with original test set',new_result,'\n')
        if os.path.exists('model_performances.txt'):
            with open('model_performances.txt', 'a') as data:
                for id, _ in enumerate(best_candidates):
                    data.write('Initial model results '+str(best_candidates[id].get('result'))+'\n')
                    data.write('Augmented results '+str(best_candidates_new[id].get('result'))+' augmentation factor '+str(augmentation_factor)+'\n')
                    data.write(str(CONF['predictive_models'][id])+' prefix_length '+str(CONF['prefix_length'])+'\n')
                data.close()
        else:
            with open('model_performances.txt', 'w') as data:
                for id, _ in enumerate(best_candidates):
                    print(best_candidates[id].get('result'))
                    data.write('Initial model results '+str(best_candidates[id].get('result'))+'\n')
                    data.write('Augmented results '+str(best_candidates_new[id].get('result'))+' augmentation factor '+str(augmentation_factor)+'\n')
                    data.write(str(CONF['predictive_models'][id])+' prefix_length '+str(CONF['prefix_length'])+'\n')
                data.close()

    logger.info('RESULT')
    #logger.info('INITIAL', initial_result)
    logger.info('Done, cheers!')

    #return {'initial_result', initial_result, 'predictive_model.config', predictive_model.config}


if __name__ == '__main__':
    dataset_list = {
        # 'hospital_billing_2',
        # 'hospital_billing_3'
         #'synthetic_data',
        #'bpi2012_W_One_TS',
        'sepsis_cases_1_start':[5, 7, 9, 11, 12, 14],
        #'BPIC15_1_f2',
        #'BPIC15_2_f2'
        #'BPIC15_3_f2',
         #'BPIC15_4_f2',
         #'BPIC15_5_f2',
        #'bpic2012_O_ACCEPTED-COMPLETE',
         #'bpic2012_O_CANCELLED-COMPLETE',
        #'bpic2012_O_DECLINED-COMPLETE',
         #'traffic_fines_1',
        #'sepsis_cases_1',
        #'sepsis_cases_2',
        #'sepsis_cases_4',
        #'legal_complaints',
         #'BPIC17_O_ACCEPTED',
    }
    for dataset,prefix_lengths in dataset_list.items():
        for prefix in prefix_lengths:
            for augmentation_factor in [0.3,0.5,0.7]:
                CONF = {  # This contains the configuration for the run
                    'data': os.path.join(dataset, 'full.xes'),
                    'train_val_test_split': [0.7, 0.15, 0.15],
                    'output': os.path.join('..', 'output_data'),
                    'prefix_length_strategy': PrefixLengthStrategy.FIXED.value,
                    'prefix_length': prefix,
                    'padding': True,  # TODO, why use of padding?
                    'feature_selection': EncodingType.COMPLEX.value,
                    'task_generation_type': TaskGenerationType.ONLY_THIS.value,
                    'attribute_encoding': EncodingTypeAttribute.LABEL.value,  # LABEL, ONEHOT
                    'labeling_type': LabelTypes.ATTRIBUTE_STRING.value,
                    'predictive_models': [ClassificationMethods.XGBOOST.value,ClassificationMethods.RANDOM_FOREST.value],  # RANDOM_FOREST, LSTM, PERCEPTRON
                    'explanator': ExplainerType.DICE_AUGMENTATION.value,
                    'augmentation_factor':augmentation_factor,# SHAP, LRP, ICE, DICE
                    'threshold': 13,
                    'top_k': 10,
                    'hyperparameter_optimisation': False,  # TODO, this parameter is not used
                    'hyperparameter_optimisation_target': HyperoptTarget.MCC.value,
                    'hyperparameter_optimisation_evaluations': 20,
                    'time_encoding': TimeEncodingType.NONE.value,
                    'target_event': None,
                    'seed': 42,
                }
                run_simple_pipeline(CONF=CONF, dataset_name=dataset)
