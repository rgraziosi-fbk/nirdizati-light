import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OneHotEncoder

PADDING_VALUE = 0
#onehot and minmaxscaler not fully done

class Encoder:
    def __init__(self, df: DataFrame = None, attribute_encoding=None,feature_selection=None,prefix_length=None):
        self.attribute_encoding = attribute_encoding
        self.feature_selection = feature_selection
        self.prefix_length = prefix_length
        self._label_encoder = {}
        self._numeric_encoder = {}
        self._label_dict = {}
        self._label_dict_decoder = {}
        self._scaled_values = {}
        self._unscaled_values = {}
        for column in df:
            if column != 'trace_id':
                if not is_numeric_dtype(df[column].dtype):#or (is_numeric_dtype(df[column].dtype) and np.any(df[column] < 0)):
                    if attribute_encoding == 'label':
                        if column == 'label':
                            self._label_encoder[column] = LabelEncoder().fit(
                                sorted(df[column].apply(lambda x: str(x))))
                            classes = self._label_encoder[column].classes_
                            transforms = self._label_encoder[column].transform(classes)
                            self._label_dict[column] = dict(zip(classes, transforms))
                            self._label_dict_decoder[column] = dict(zip(transforms, classes))
                        else:
                            self._label_encoder[column] = LabelEncoder().fit(
                                sorted(pd.concat([pd.Series([str(PADDING_VALUE)]), df[column].apply(lambda x: str(x))])))
                            classes = self._label_encoder[column].classes_
                            transforms = self._label_encoder[column].transform(classes)
                            self._label_dict[column] = dict(zip(classes, transforms))
                            self._label_dict_decoder[column] = dict(zip(transforms, classes))
                    elif attribute_encoding == "onehot":
                        if column == 'label':
                            self._label_encoder[column] = LabelEncoder().fit(
                                sorted(df[column].apply(lambda x: str(x))))
                            classes = self._label_encoder[column].classes_
                            transforms = self._label_encoder[column].transform(classes)
                            self._label_dict[column] = dict(zip(classes, transforms))
                            self._label_dict_decoder[column] = dict(zip(transforms, classes))
                        else:
                            self._label_encoder[column] = OneHotEncoder(drop='if_binary', sparse_output=False,
                                       handle_unknown='ignore').fit(df[column].astype(str).values.reshape(-1,1))
                            categories = self._label_encoder[column].categories_[0].reshape(-1, 1)
                            transforms = [tuple(enc) for enc in self._label_encoder[column].transform(categories)]
                            classes = list(categories.flatten())
                            self._label_dict[column] = dict(zip(classes, transforms))
                            self._label_dict_decoder[column] = dict(zip(transforms, classes))

                else:
                    # Assuming df[column] contains your timestamps
                    unscaled_values = df[column].values

                    # Exclude 0 values from the MinMaxScaler
                    non_zero_values = unscaled_values[unscaled_values != 0.0].reshape(-1,1)
                    scaler = MinMaxScaler()

                    # Check if all values in the column are 0.0
                    if non_zero_values.size == 0:

                        # Handle case where all values are 0.0 (padding)
                        scaled_values = unscaled_values  # Padding values remain unchanged
                        self._numeric_encoder[column] = scaler.fit(unscaled_values.reshape(-1,1))
                        scaled_values_full = scaled_values# No scaler needed for all 0 values
                    else:
                        # Use MinMaxScaler for non-padding timestamps
                        self._numeric_encoder[column] = scaler.fit(non_zero_values)
                        # Scale the non-padding timestamps
                        scaled_values = self._numeric_encoder[column].transform(non_zero_values).flatten()

                        # Initialize scaled values with padding timestamps included
                        scaled_values_full = np.zeros_like(unscaled_values)
                        scaled_values_full[unscaled_values != 0.0] = scaled_values

                    # Update your _scaled_values and _unscaled_values dictionaries
                    self._scaled_values[column] = scaled_values_full
                    self._unscaled_values[column] = unscaled_values

                    # Print information
                    print('column:', column, 'considered number, top 5 values are:', list(df[column][:5]))

    def encode(self, df: DataFrame) -> None:
        for column in df:
            if column != 'trace_id':
                if column in self._label_encoder:
                    df[column] = df[column].apply(lambda x: self._label_dict[column].get(str(x), PADDING_VALUE))
                else:
                    non_padding_mask = df[column] != 0
                    if non_padding_mask.any():  # Check if there are non-padding values
                        non_padding_values = df[column][non_padding_mask].values.reshape(-1, 1)
                        transformed_values = self._numeric_encoder[column].transform(non_padding_values).flatten()
                        df.loc[non_padding_mask, column] = transformed_values

    def decode(self, df: DataFrame) -> None:
        for column in df:
                if column != 'trace_id' and '.' not in column:
                    if column in self._label_encoder:
                        df[column] = df[column].apply(lambda x: self._label_dict_decoder[column].get(x, PADDING_VALUE))
                    else:
                        df[column] = self._numeric_encoder[column].inverse_transform(df[column].values.reshape(-1,1)).flatten()

    def decode_row(self, row) -> np.array:
        decoded_row = []
        for column, value in row.items():
            if column != 'trace_id':
                if column in self._label_encoder:
                     decoded_row += [self._label_dict_decoder[column].get(value, PADDING_VALUE)]
                elif column in self._numeric_encoder:
                    decoded_row += [self._numeric_encoder[column].inverse_transform(np.array(value).reshape(-1,1))[0][0]]
            else:
                decoded_row += [value]
        return np.array(decoded_row)

    def decode_column(self, column, column_name) -> np.array:
        decoded_column = []
        if column != 'trace_id':
            if column_name in self._encoder:
                if not is_numeric_dtype(df[column].dtype):
                    decoded_column += [self._label_dict_decoder[column_name].get(x, PADDING_VALUE) for x in column]
                else:
                    decoded_column += [self._unscaled_values[column_name].get(x) for x in column]
        else:
            decoded_column += list(column)
        return np.array(decoded_column)

    def get_values(self, column_name):
        if not is_numeric_dtype(df[column].dtype):
            return (self._label_dict[column_name].keys(), self._label_dict_decoder[column_name].keys())
