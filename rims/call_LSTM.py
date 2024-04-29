
import json
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from keras.models import load_model
import rims.entities as en


from pickle import load
from enum import Enum

import warnings
warnings.filterwarnings("ignore")


class InstanceState(Enum):
    WAITING = 1
    INEXECUTION = 2
    COMPLETE = 3


class Predictor(object):

    def __init__(self, model_path, params):
        """constructor"""
        self.g1, self.first_session, self.proc_model_path = self._load_models(model_path[0])
        self.g2, self.second_session, self.wait_model_path = self._load_models(model_path[1])
        self.parms = {'update_gen': False, 'update_ia_gen': False, 'update_mpdf_gen': False, 'update_times_gen': False, 'save_models': True, 'evaluate': True, 'mining_alg': 'sm1', 'read_options': {'timeformat': '%Y-%m-%dT%H:%M:%S.%f', 'column_names': {'Case ID': 'caseid', 'Activity': 'task', 'lifecycle:transition': 'event_type', 'Resource': 'user', 'Start Timestamp': 'start_timestamp', 'Complete Timestamp': 'end_timestamp'}, 'one_timestamp': False, 'filter_d_attrib': True}, 'event_logs_path': 'input_files/event_logs', 'bpmn_models': 'input_files/bpmn_models', 'embedded_path': 'input_files/embedded_matix', 'ia_gen_path': 'input_files/ia_gen_models', 'seq_flow_gen_path': 'input_files/seq_flow_gen_models', 'times_gen_path': 'input_files/times_gen_models', 'sm1_path': 'external_tools/splitminer/splitminer.jar', 'sm2_path': 'external_tools/splitminer2/sm2.jar', 'sm3_path': 'external_tools/splitminer3/bpmtk.jar', 'bimp_path': 'external_tools/bimp/qbp-simulator-engine.jar', 'align_path': 'external_tools/proconformance/ProConformance2.jar', 'calender_path': 'external_tools/calenderimp/CalenderImp.jar', 'simulator': 'bimp', 'sim_metric': 'tsd', 'add_metrics': ['day_hour_emd', 'log_mae', 'dl', 'mae'], 'exp_reps': 1, 'imp': 1, 'max_eval': 12, 'batch_size': 32, 'epochs': 200, 'n_size': [5, 10, 15], 'l_size': [50, 100], 'lstm_act': ['selu', 'tanh'], 'dense_act': ['linear'], 'optim': ['Nadam'], 'model_type': 'dual_inter', 'opt_method': 'bayesian', 'all_r_pool': True, 'reschedule': False, 'rp_similarity': 0.8}
        self.n_feat_proc = (self.proc_model_path.get_layer('features').output_shape[0][2])
        self.n_feat_wait = (self.wait_model_path.get_layer('features').output_shape[0][2])
        self.params = params
        self.active_instances = dict()

    def _load_models(self, path):
        graph = tf.Graph()
        with graph.as_default():
            session = tf.compat.v1.Session()
            with session.as_default():
                model = load_model(path)
        return graph, session, model

    def predict(self):
        if os.path.exists(self.params.METADATA):
            with open(self.params.METADATA) as file:
                data = json.load(file)
                self.ac_index = data['ac_index']
                self.index_ac = {v: k for k, v in self.ac_index.items()}
                self.n_size = data['n_size']
                rl_table = pd.DataFrame([{'role_name': item[0],
                                          'size': len(item[1]),
                                          'role_index': num}
                                         for num, item in enumerate(data['roles'].items())])
                pr_act_initial = int(
                    round(data['inter_mean_states']['wip']))
                init_states = data['inter_mean_states']['tasks']
        self.scaler = load(open(self.params.SCALER, 'rb'))
        self.inter_scaler = load(open(self.params.INTER_SCALER, 'rb'))
        self.end_inter_scaler = load(open(self.params.END_INTER_SCALER, 'rb'))
        self.rl_dict = self._initialize_roles(rl_table, check_avail=self.parms['reschedule'])
        self.ac_dict = self._initialize_activities(self.ac_index, init_states)


    # transition =  = (3, 0), rp_oc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ac_wip = 1,  time = datetime(year=2012, month=3, day=13, hour=3, minute=49, second=0)
    def processing_time(self, cid, pr_wip, transition, ac_wip, rp_oc, time, queue):
        # initialize instance
        if cid not in self.active_instances:
            self.active_instances[cid] = en.ProcessInstance(cid, self.n_size, (self.n_feat_proc, self.n_feat_wait), dual=True, n_act=True)
        wip = self.inter_scaler.transform(np.array([pr_wip, ac_wip]).reshape(-1, 2))[0]
        self.active_instances[cid].update_proc_ngram(transition[0], time, wip, rp_oc)
        act_ngram, feat_ngram = self.active_instances[cid].get_proc_ngram()
        # predict
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            with self.g1.as_default():
                with self.first_session.as_default():
                    preds = self.proc_model_path.predict(
                        {'ac_input': np.array([act_ngram]),
                         'features': feat_ngram})
        preds[preds < 0] = 0.000001
        proc_t = preds[0]
        self.active_instances[cid].update_proc(proc_t)
        ipred = self.scaler.inverse_transform(np.concatenate((preds, np.array([[0.000]])), axis=1))
        iproc_t = ipred[0][0]
        # resource assignment
        #release_time = time + timedelta(seconds=int(round(iproc_t)))

        return int(round(iproc_t))

    # pr_act_initial= 11, next_act = (7, 3), rp_oc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], time = datetime(year=2012, month=3, day=13, hour=5, minute=28, second=26)
    def predict_waiting_queue(self, cid, pr_wip, next_act, rp_oc, time, queue):
        # initialize instance
        if cid not in self.active_instances:
            self.active_instances[cid] = en.ProcessInstance(cid, self.n_size, (self.n_feat_proc, self.n_feat_wait), dual=True, n_act=True)
        wip = self.end_inter_scaler.transform(np.array([pr_wip, queue]).reshape(-1, 2))[0]
        self.active_instances[cid].update_wait_ngram(next_act[0], time, wip, rp_oc)
        n_act_ngram, feat_ngram = self.active_instances[cid].get_wait_ngram()
        # predict
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            with self.g2.as_default():
                with self.second_session.as_default():
                    preds = self.wait_model_path.predict(
                        {'ac_input': np.array([n_act_ngram]),
                         'features': feat_ngram})
        preds[preds < 0] = 0.000001
        wait_t = preds[0]
        self.active_instances[cid].update_wait(wait_t)
        ipred = self.scaler.inverse_transform(np.concatenate((np.array([[0.000]]), preds), axis=1))
        iwait_t = ipred[0][1]

        return int(iwait_t)


    def predict_waiting(self, cid, pr_wip, next_act, rp_oc, time, queue):
        # initialize instance
        if cid not in self.active_instances:
            self.active_instances[cid] = en.ProcessInstance(cid, self.n_size, (self.n_feat_proc, self.n_feat_wait), dual=True, n_act=True)
        wip = self.end_inter_scaler.transform(np.array([pr_wip]).reshape(-1, 1))[0]
        self.active_instances[cid].update_wait_ngram(next_act[0], time, wip, rp_oc)
        n_act_ngram, feat_ngram = self.active_instances[cid].get_wait_ngram()
        # predict
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            with self.g2.as_default():
                with self.second_session.as_default():
                    preds = self.wait_model_path.predict(
                        {'ac_input': np.array([n_act_ngram]),
                         'features': feat_ngram})
        preds[preds < 0] = 0.000001
        wait_t = preds[0]
        self.active_instances[cid].update_wait(wait_t)
        ipred = self.scaler.inverse_transform(np.concatenate((np.array([[0.000]]), preds), axis=1))
        iwait_t = ipred[0][1]

        return int(iwait_t)

    @staticmethod
    def _initialize_activities(ac_index, init_states):
        activities_dict = dict()
        for key, value in ac_index.items():
            if key not in ['Start', 'End']:
                activities_dict[value] = en.ActivityCounter(
                    key, index=value, initial=int(round(init_states.get(key, 0))))
        return activities_dict

    @staticmethod
    def _initialize_roles(roles, check_avail):
        rl_dict = dict()
        for role in roles.to_dict('records'):
            rl_dict[role['role_index']] = en.Role(role['role_name'],
                                                  role['size'],
                                                  index=role['role_index'],
                                                  check_avail=check_avail)
        return rl_dict

    @staticmethod
    def _initialize_queue(iarr):
        queue = en.Queue()
        for k, v in iarr.items():
            queue.add({'timestamp': v,
                       'action': 'create_instance',
                       'caseid': k})
        return queue

    @staticmethod
    def _initialize_exec_state(sequences):
        execution_state = dict()
        for k, transitions in sequences.items():
            execution_state[k] = {'state': InstanceState.WAITING,
                                  'transitions': transitions}
        return execution_state

    @staticmethod
    def _encode_secuences(sequences, ac_idx, rl_task, rl_table):
        seq = sequences.copy()
        # Determine biggest resource pool as default
        def_role = rl_table[rl_table['size'] == rl_table['size'].max()].iloc[0]['role_name']
        # Assign roles to activities
        seq['ac_index'] = seq.apply(
            lambda x: ac_idx[x.task], axis=1)
        seq = seq.merge(rl_task,
                        how='left',
                        on='task')
        seq.fillna(value={'role': def_role}, inplace=True)
        seq = seq.merge(rl_table,
                        how='left',
                        left_on='role', right_on='role_name')
        ac_rl = lambda x: (x.ac_index, x.role_index)
        seq['ac_rl'] = seq.apply(ac_rl, axis=1)
        num_elements = 0
        encoded_seq = dict()
        for key, group in seq.sort_values('pos_trace').groupby('caseid'):
            encoded_seq[key] = group.ac_rl.to_list()
            num_elements += len(encoded_seq[key])
        return encoded_seq, num_elements
