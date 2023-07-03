import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

PADDING_VALUE = '0'


class Encoder:
    def __init__(self, df: DataFrame = None, attribute_encoding=None):
        self.attribute_encoding = attribute_encoding
        self._encoder = {}
        self._label_dict = {}
        self._label_dict_decoder = {}
        for column in df:

            if column != 'trace_id':
                if df[column].dtype != int or (df[column].dtype == int and np.any(df[column] < 0)):

                    if attribute_encoding == "label":
                        self._encoder[column] = LabelEncoder().fit(
                            sorted(pd.concat([pd.Series([str(PADDING_VALUE)]), df[column].apply(lambda x: str(x))])))
                        classes = self._encoder[column].classes_
                        transforms = self._encoder[column].transform(classes)
                        self._label_dict[column] = dict(zip(classes, transforms))
                        self._label_dict_decoder[column] = dict(zip(transforms, classes))

                    elif attribute_encoding == "onehot":
                        padded_values = pd.concat([pd.Series([str(PADDING_VALUE)]), df[column].apply(lambda x: str(x))])
                        label_enc = pd.DataFrame(LabelEncoder().fit_transform(sorted(padded_values)))
                        self._encoder[column] = OneHotEncoder(sparse=False).fit(label_enc)
                        categories = self._encoder[column].categories_[0].reshape(-1, 1)
                        transforms = [tuple(enc) for enc in self._encoder[column].transform(categories)]
                        classes = list(padded_values.unique())
                        self._label_dict[column] = dict(zip(classes, transforms))
                        self._label_dict_decoder[column] = dict(zip(transforms, classes))

                    else:
                        pass

    def encode(self, df: DataFrame) -> None:
        for column in df:
            if column in self._encoder:
                df[column] = df[column].apply(lambda x: self._label_dict[column].get(str(x), PADDING_VALUE))

    def decode(self, df: DataFrame) -> None:
        for column in df:
            if column in self._encoder:
                df[column] = df[column].apply(lambda x: self._label_dict_decoder[column].get(x, PADDING_VALUE))

    def decode_row(self, row) -> np.array:
        decoded_row = []
        for column, value in row.iteritems():
            if column in self._encoder:
                decoded_row += [self._label_dict_decoder[column].get(value, PADDING_VALUE)]
            else:
                decoded_row += [value]
        return np.array(decoded_row)

    def decode_column(self, column, column_name) -> np.array:
        decoded_column = []
        if column_name in self._encoder:
            decoded_column += [self._label_dict_decoder[column_name].get(x, PADDING_VALUE) for x in column]
        else:
            decoded_column += list(column)
        return np.array(decoded_column)

    def get_values(self, column_name):
        return (self._label_dict[column_name].keys(), self._label_dict_decoder[column_name].keys())
