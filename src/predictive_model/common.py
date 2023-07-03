from enum import Enum
from pandas import DataFrame
import numpy as np
from funcy import flatten


class ClassificationMethods(Enum):
    RANDOM_FOREST = 'randomForestClassifier'
    KNN = 'knn'
    XGBOOST = 'xgboost'
    SGDCLASSIFIER = 'SGDClassifier'
    PERCEPTRON = 'perceptron'
    LSTM = 'lstm'
    MLP = 'mlp'
    SVM = 'svc'


class RegressionMethods(Enum):
    RANDOM_FOREST = 'randomForestRegressor'


def get_tensor(CONF, df: DataFrame):

    trace_attributes = [att for att in df.columns if '_' not in att]
    event_attributes = [att[:-2] for att in df.columns if att[-2:] == '_1']

    reshaped_data = {
            trace_index: {
                prefix_index:
                    list(flatten(
                        feat_values if isinstance(feat_values, tuple) else [feat_values]
                        for feat_name, feat_values in trace.iteritems()
                        if feat_name in trace_attributes + [event_attribute + '_' + str(prefix_index) for event_attribute in event_attributes]
                    ))
                for prefix_index in range(1, CONF['prefix_length'] + 1)
            }
            for trace_index, trace in df.iterrows()
    }

    flattened_features = max(
        len(reshaped_data[trace][prefix])
        for trace in reshaped_data
        for prefix in reshaped_data[trace]
    )

    tensor = np.zeros((
        len(df),                # sample
        CONF['prefix_length'],  # time steps
        flattened_features      # features x single time step (trace and event attributes)
    ))

    for trace_index in reshaped_data:  # prefix
        for prefix_index in reshaped_data[trace_index]:  # steps of the prefix
            for single_flattened_value in range(len(reshaped_data[trace_index][prefix_index])):
                tensor[trace_index, prefix_index - 1, single_flattened_value] = reshaped_data[trace_index][prefix_index][single_flattened_value]

    return tensor

def shape_label_df(df: DataFrame):

    labels_list = df['label'].tolist()
    labels = np.zeros((len(labels_list), int(max(df['label'].nunique(), int(max(df['label'].values))) +1)))
    for label_idx, label_val in enumerate(labels_list):
        labels[int(label_idx), int(label_val)] = 1

    return labels