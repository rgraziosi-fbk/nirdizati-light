import logging
import numpy as np
import torch
from hyperopt import STATUS_OK, STATUS_FAIL
from pandas import DataFrame
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier,XGBRegressor
from nirdizati_light.evaluation.common import evaluate_classifier, evaluate_regressor
from nirdizati_light.predictive_model.common import ClassificationMethods, RegressionMethods, get_tensor, shape_label_df, LambdaModule, EarlyStopper

logger = logging.getLogger(__name__)


def drop_columns(df: DataFrame) -> DataFrame:
    df = df.drop(['trace_id', 'label'],axis=1)
    return df

class PredictiveModel:
    """
    A class representing a predictive model

    :param dict CONF: configuration dictionary (the only required attribute is 'prefix_length')
    :param nirdizati_light.predictive_model.common.ClassificationMethods model_type: type of predictive model
    :param pandas.DataFrame train_df: training data to train model
    :param pandas.DataFrame validate_df: validation data to evaluate model
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
            self.model = self._instantiate_model(config)
            self._fit_model(self.model, config)
            actual = self.full_validate_df['label']
            
            if self.model_type is ClassificationMethods.LSTM.value:
                actual = np.array(actual.to_list())

            if self.model_type in [item.value for item in ClassificationMethods]:
                predicted, scores = self.predict()
                result = evaluate_classifier(actual, predicted, scores, loss=target)
            elif self.model_type in [item.value for item in RegressionMethods]:
                predicted = self.model.predict(self.validate_df)
                result = evaluate_regressor(actual, predicted, loss=target)
            else:
                raise Exception('Unsupported model_type')

            return {
                'status': STATUS_OK,
                'loss': - result['loss'],  # we are using fmin for hyperopt
                'exception': None,
                'config': config,
                'model': self.model,
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
        elif self.model_type is ClassificationMethods.DT.value:
            model = DecisionTreeClassifier(**config)
        elif self.model_type == ClassificationMethods.KNN.value:
            model = KNeighborsClassifier(**config)
        elif self.model_type == ClassificationMethods.XGBOOST.value:
            model = XGBClassifier(**config)
        elif self.model_type == ClassificationMethods.SGDCLASSIFIER.value:
            model = SGDClassifier(**config)
        elif self.model_type == ClassificationMethods.PERCEPTRON.value:
            # added CalibratedClassifier to get predict_proba from perceptron model
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
            model = torch.nn.Sequential(
                torch.nn.LSTM(
                    input_size=self.train_tensor.shape[2],
                    hidden_size=int(config['lstm_hidden_size']),
                    num_layers=int(config['lstm_num_layers']),
                    batch_first=True
                ),
                LambdaModule(lambda x: x[0][:,-1,:]),
                torch.nn.Linear(int(config['lstm_hidden_size']), self.train_label.shape[1]),
                torch.nn.Softmax(dim=1),
            ).to(torch.float32)
        else:
            raise Exception('unsupported model_type')
        
        return model

    def _fit_model(self, model, config=None):
        if self.model_type is ClassificationMethods.LSTM.value:
            MAX_NUM_EPOCHS = 500

            train_tensor = torch.tensor(self.train_tensor, dtype=torch.float32)
            validate_tensor = torch.tensor(self.validate_tensor, dtype=torch.float32)

            early_stopper = EarlyStopper(patience=5, min_delta=0.01)

            for _ in range(MAX_NUM_EPOCHS):
                # training
                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
                criterion = torch.nn.CrossEntropyLoss()
                optimizer.zero_grad()
                output = model(train_tensor)
                loss = criterion(output, torch.tensor(self.train_label, dtype=torch.float32))
                loss.backward()
                optimizer.step()
                
                # validation
                model.eval()
                validate_loss = criterion(model(validate_tensor), torch.tensor(self.validate_label, dtype=torch.float32))
                if early_stopper.early_stop(validate_loss):             
                    break

        elif self.model_type not in (ClassificationMethods.LSTM.value):
            model.fit(self.train_df.values, self.full_train_df['label'])

    def predict(self):
        """
        Perform predictions with the model and return them
        """
        if self.model_type is ClassificationMethods.LSTM.value:
            validate_tensor = torch.tensor(self.validate_tensor, dtype=torch.float32)

            probabilities = self.model(validate_tensor).detach().numpy()
            predicted = np.argmax(probabilities, axis=1)
            scores = np.amax(probabilities, axis=1)
        elif self.model_type not in (ClassificationMethods.LSTM.value):
            predicted = self.model.predict(self.validate_df)
            scores = self.model.predict_proba(self.validate_df)[:, 1]
        else:
            raise Exception('Unsupported model_type')

        return predicted, scores
