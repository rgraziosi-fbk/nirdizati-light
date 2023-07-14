import logging
from enum import Enum
from operator import itemgetter

from pandas import DataFrame

from nirdizati_light.hyperparameter_optimisation.common import HyperoptTarget, retrieve_best_model
from nirdizati_light.predictive_model.predictive_model import PredictiveModel

logger = logging.getLogger(__name__)


def drop_columns(df: DataFrame) -> DataFrame:
    df = df.drop(['trace_id', 'label'], 1)
    return df


class RecommenderModelInstantiation(Enum):
    CLASSIFIER = 'classifier'
    REGRESSOR = 'regressor'


class ClassifierAndRegressor:

    def __init__(self, model_type, CLASSIFIER_CONF, classifier_train_df, classifier_validate_df, REGRESSOR_CONF, regressor_train_df, regressor_validate_df):
        self.model_type = model_type
        self.full_classifier_train_df = classifier_train_df
        self.full_classifier_validate_df = classifier_validate_df

        self.full_regressor_train_df = regressor_train_df
        self.full_regressor_validate_df = regressor_validate_df
        self.model = dict()
        self.model[RecommenderModelInstantiation.CLASSIFIER.value] = PredictiveModel(
            CLASSIFIER_CONF,
            CLASSIFIER_CONF['predictive_model'],
            self.full_classifier_train_df,
            self.full_classifier_validate_df
        )
        self.model[RecommenderModelInstantiation.REGRESSOR.value] = PredictiveModel(
            REGRESSOR_CONF,
            REGRESSOR_CONF['predictive_model'],
            self.full_regressor_train_df,
            self.full_regressor_validate_df
        )

    def fit(self, max_evaluations, target=dict({
            RecommenderModelInstantiation.CLASSIFIER.value: HyperoptTarget.AUC.value,
            RecommenderModelInstantiation.REGRESSOR.value: HyperoptTarget.MAE.value
        })):
        # ottimizza classificatore
        print('Optimizing the classifier')
        self.model[RecommenderModelInstantiation.CLASSIFIER.value].model, \
        self.model[RecommenderModelInstantiation.CLASSIFIER.value].config = retrieve_best_model(
            self.model[RecommenderModelInstantiation.CLASSIFIER.value],
            self.model[RecommenderModelInstantiation.CLASSIFIER.value].model_type,
            max_evaluations,
            target[RecommenderModelInstantiation.CLASSIFIER.value]
        )
        # ottimizza regressore
        print('Optimizing the regressor')
        self.model[RecommenderModelInstantiation.REGRESSOR.value].model, \
        self.model[RecommenderModelInstantiation.REGRESSOR.value].config = retrieve_best_model(
            self.model[RecommenderModelInstantiation.REGRESSOR.value],
            self.model[RecommenderModelInstantiation.REGRESSOR.value].model_type,
            max_evaluations,
            target[RecommenderModelInstantiation.REGRESSOR.value]
        )

    def recommend(self, df, top_n):
        recommendations = []
        for trace, likely_next in zip(df.T,
                               self.model[RecommenderModelInstantiation.CLASSIFIER.value].model.predict_proba(
                                       drop_columns(df))):
            likely_next_activities = likely_next.argsort()[-top_n:][::-1]
            ranked_next_activities = [
                (
                    next_activity,
                    self.model[RecommenderModelInstantiation.REGRESSOR.value].model.predict(
                        [list(drop_columns(df).T[trace].values.reshape(1, -1)[0]) + [next_activity]])
                )
                for next_activity in likely_next_activities
            ]

            recommendations += [ min(ranked_next_activities, key=itemgetter(1))[0] ]
        return recommendations

