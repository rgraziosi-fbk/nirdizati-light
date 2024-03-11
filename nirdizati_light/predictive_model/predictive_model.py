import logging

# import tensorflow as tf
import numpy as np
# import tensorflow.python.keras.models
from hyperopt import STATUS_OK, STATUS_FAIL
from pandas import DataFrame
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from nirdizati_light.evaluation.common import evaluate_classifier, evaluate_regressor
from nirdizati_light.predictive_model.common import ClassificationMethods, RegressionMethods, get_tensor, shape_label_df

logger = logging.getLogger(__name__)


def drop_columns(df: DataFrame) -> DataFrame:
    df = df.drop(['trace_id', 'label'],axis=1)
    return df

class PredictiveModel:
    """
    A class representing a predictive model

    Constructor parameters:
        CONF: configuration dictionary (the only required attribute is 'prefix_length')
        model_type: type of predictive model (can be one of predictive_model.common.ClassificationMethods)
        train_df: training data to train model as a pandas dataframe
        validate_df: validation data to evaluate model as a pandas dataframe
    """

    def __init__(self, CONF, model_type, train_df, validate_df):
        self.model_type = model_type
        self.config = None
        self.model = None
        self.full_train_df = train_df
        self.train_df = drop_columns(train_df)
        self.train_df_shaped = None
        self.full_validate_df = validate_df
        self.validate_df = drop_columns(validate_df)
        self.validate_df_shaped = None

        if model_type is ClassificationMethods.LSTM.value:
            self.train_tensor = get_tensor(CONF, self.train_df)
            self.validate_tensor = get_tensor(CONF, self.validate_df)
            self.train_label = shape_label_df(self.full_train_df)
            self.validate_label = shape_label_df(self.full_validate_df)
        elif model_type is ClassificationMethods.MLP.value:
            self.train_label = self.full_train_df['label'].nunique()
            self.validate_label = self.full_validate_df['label'].nunique()
    
    def train_and_evaluate_configuration(self, config, target):
        try:
            model = self._instantiate_model(config)
            self._fit_model(model)
            actual = self.full_validate_df['label']
            
            if self.model_type is ClassificationMethods.LSTM.value:
                actual = np.array(actual.to_list())

            if self.model_type in [item.value for item in ClassificationMethods]:
                predicted, scores = self._output_model(model=model)
                result = evaluate_classifier(actual, predicted, scores, loss=target)
            elif self.model_type in [item.value for item in RegressionMethods]:
                predicted = model.predict(self.validate_df)
                result = evaluate_regressor(actual, predicted, loss=target)
            else:
                raise Exception('Unsupported model_type')

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
        if self.model_type is ClassificationMethods.RANDOM_FOREST.value:
            model = RandomForestClassifier(**config)
        elif self.model_type == ClassificationMethods.KNN.value:
            model = KNeighborsClassifier(**config)
        elif self.model_type == ClassificationMethods.XGBOOST.value:
            model = XGBClassifier(**config)
        elif self.model_type == ClassificationMethods.SGDCLASSIFIER.value:
            model = SGDClassifier(**config)
        elif self.model_type == ClassificationMethods.PERCEPTRON.value:
            #added CalibratedClassifier to get predict_proba from perceptron model
            model = Perceptron(**config)
            model = CalibratedClassifierCV(model, cv=10, method='isotonic')
        elif self.model_type is ClassificationMethods.MLP.value:
            model = MLPClassifier(**config)
            #model = CalibratedClassifierCV(model, cv=10, method='isotonic')
        elif self.model_type == RegressionMethods.RANDOM_FOREST.value:
            model = RandomForestRegressor(**config)
        elif self.model_type == ClassificationMethods.SVM.value:
            model = SVC(**config,probability=True)
        elif self.model_type is ClassificationMethods.LSTM.value:
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

    def _fit_model(self, model,config=None):
        if self.model_type is ClassificationMethods.LSTM.value:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0,
                                                              mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

            model.fit(self.train_tensor, {'output': self.train_label},
                      validation_split=0.1,
                      verbose=1,
                      callbacks=[early_stopping, lr_reducer],
                      batch_size=64,
                      epochs=1)
        elif self.model_type not in (ClassificationMethods.LSTM.value):
            model.fit(self.train_df.values, self.full_train_df['label'])

    def _output_model(self, model):
        if self.model_type is ClassificationMethods.LSTM.value:
            probabilities = model.predict(self.validate_tensor)
            predicted = np.argmax(probabilities, axis=1)
            scores = np.amax(probabilities, axis=1)
        elif self.model_type not in (ClassificationMethods.LSTM.value):
            predicted = model.predict(self.validate_df)
            scores = model.predict_proba(self.validate_df)[:, 1]
        else:
            raise Exception('Unsupported model_type')

        return predicted, scores
