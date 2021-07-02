from enum import Enum
from pandas import DataFrame
import numpy as np


class ClassificationMethods(Enum):
    RANDOM_FOREST = 'randomForestClassifier'
    KNN = 'knn'
    XGBOOST = 'xgboost'
    SGDCLASSIFIER = 'SGDClassifier'
    PERCEPTRON = 'perceptron'
    LSTM = 'lstm'


class RegressionMethods(Enum):
    RANDOM_FOREST = 'randomForestRegressor'


def get_tensor(CONF, df: DataFrame):

    tuple_lists = df.to_numpy().tolist()
    max_enc_length = max([len(onehot_enc) for onehot_enc in tuple_lists[0]])

    tensor = np.zeros((len(tuple_lists),
                       CONF['prefix_length'],
                       max_enc_length))

    for i_prefix, prefix in enumerate(tuple_lists):
        for i_tuple_enc, tuple_enc in enumerate(prefix):
            for i_enc_value, enc_value in enumerate(tuple_enc):
                tensor[i_prefix, i_tuple_enc, i_enc_value] = enc_value

    return tensor


def shape_label_df(df: DataFrame):

    labels_list = df['label'].tolist()
    labels = np.zeros((len(labels_list),
                       len(labels_list[0])))

    for i_label_enc, label_enc in enumerate(labels_list):
        for i_enc_value, enc_value in enumerate(label_enc):
            labels[i_label_enc, i_enc_value] = enc_value

    return labels
