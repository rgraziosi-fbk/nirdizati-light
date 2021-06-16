import logging

from hyperopt import STATUS_OK, STATUS_FAIL
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import numpy as np
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from src.evaluation.common import evaluate
from src.predictive_model.common import PredictionMethods, get_tensor, shape_label_df

logger = logging.getLogger(__name__)


def drop_columns(df: DataFrame) -> DataFrame:
    df = df.drop(['trace_id', 'label'], 1)
    return df

class PredictiveModel:

    def __init__(self, CONF, model_type, train_df, validate_df):
        self.CONF = CONF
        self.model_type = model_type
        self.config = None
        self.model = None
        self.full_train_df = train_df
        self.train_df = drop_columns(train_df)
        self.train_df_shaped = None
        self.full_validate_df = validate_df
        self.validate_df = drop_columns(validate_df)
        self.validate_df_shaped = None

        if model_type is PredictionMethods.LSTM.value:
            self.train_tensor = get_tensor(CONF, self.train_df)
            self.validate_tensor = get_tensor(CONF, self.validate_df)
            self.train_label = shape_label_df(self.full_train_df)
            self.validate_label = shape_label_df(self.full_validate_df)

    def train_and_evaluate_configuration(self, config, target):
        try:
            model = self._instantiate_model(config)

            self._fit_model(model)

            predicted, scores = self._output_model(model=model)

            actual = self.full_validate_df['label']
            if self.CONF['predictive_model'] is PredictionMethods.LSTM.value:
                actual = np.argmax(np.array(actual.to_list()), axis=1)

            result = evaluate(actual, predicted, scores, loss=target)

            return {
                'status': STATUS_OK,
                'loss': - result['loss'],  # we are using fmin for hyperopt
                'exception': None,
                'config': config,
                'model': model,
                'result': result,
            }
        except Exception as e:
            return {
                'status': STATUS_FAIL,
                'loss': 0,
                'exception': str(e)
            }

    def _instantiate_model(self, config):
        if self.model_type is PredictionMethods.RANDOM_FOREST.value:
            model = RandomForestClassifier(**config)
        elif self.model_type == PredictionMethods.KNN.value:
            model = KNeighborsClassifier(**config)
        elif self.model_type == PredictionMethods.XGBOOST.value:
            model = XGBClassifier(**config)
        elif self.model_type == PredictionMethods.SGDCLASSIFIER.value:
            model = SGDClassifier(**config)
        elif self.model_type == PredictionMethods.PERCEPTRON.value:
            model = Perceptron(**config)

        elif self.model_type is PredictionMethods.LSTM.value:
            # input layer
            main_input = tf.keras.layers.Input(shape=(self.train_tensor.shape[1], self.train_tensor.shape[2]),
                                               name='main_input')

            # hidden layer
            b1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100,
                                                                    use_bias=True,
                                                                    implementation=1,
                                                                    activation=config['activation'],
                                                                    kernel_initializer=config['kernel_initializer'],
                                                                    return_sequences=False,
                                                                    dropout=0.2))(main_input)

            # output layer
            output = tf.keras.layers.Dense(self.train_label.shape[1],
                                           activation='softmax',
                                           name='output',
                                           kernel_initializer=config['kernel_initializer'])(b1)

            model = tf.keras.models.Model(inputs=[main_input], outputs=[output])
            model.compile(loss={'output': 'categorical_crossentropy'}, optimizer=config['optimizer'])
            model.summary()

        else:
            raise Exception('unsupported model_type')
        return model

    def _fit_model(self, model):

        if self.model_type is not PredictionMethods.LSTM.value:
            model.fit(self.train_df, self.full_train_df['label'])

        elif self.model_type is PredictionMethods.LSTM.value:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0,
                                                              mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

            model.fit(self.train_tensor, {'output': self.train_label},
                      validation_split=0.1,
                      verbose=1,
                      callbacks=[early_stopping, lr_reducer],
                      batch_size=128,
                      epochs=1)


    def _output_model(self, model):

        if self.model_type is not PredictionMethods.LSTM.value:
            predicted = model.predict(self.validate_df)
            scores = model.predict_proba(self.validate_df)[:, 1]
        elif self.model_type is PredictionMethods.LSTM.value:
            probabilities = model.predict(self.validate_tensor)
            predicted = np.argmax(probabilities, axis=1)
            scores = np.amax(probabilities, axis=1)
        else:
            raise Exception('unsupported model_type')

        return predicted, scores
