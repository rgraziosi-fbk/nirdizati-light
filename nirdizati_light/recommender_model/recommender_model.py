from nirdizati_light.recommender_model.classifier_and_regressor.classifier_and_regressor import ClassifierAndRegressor
from nirdizati_light.recommender_model.common import RecommendationMethods


class RecommenderModel:
    def __init__(self,
                 CONF,
                 CONF_CLASSIFIER, classifier_train_df, classifier_validate_df,
                 CONF_REGRESSOR, regressor_train_df, regressor_validate_df):
        self.model_type = CONF['recommender_model']
        self.model = None
        self.CONF_CLASSIFIER = CONF_CLASSIFIER
        self.classifier_train_df = classifier_train_df
        self.classifier_validate_df = classifier_validate_df
        self.CONF_REGRESSOR = CONF_REGRESSOR
        self.regressor_train_df = regressor_train_df
        self.regressor_validate_df = regressor_validate_df
        self._instantiate_model()

    def retrieve_best_model(self, max_evaluations, target):
        self.model.fit(max_evaluations, target)

    def _instantiate_model(self):
        if self.model_type is RecommendationMethods.CLASSIFICATION_AND_REGRESSION.value:
            self.model = ClassifierAndRegressor(
                self.model_type,
                self.CONF_CLASSIFIER,
                self.classifier_train_df,
                self.classifier_validate_df,
                self.CONF_REGRESSOR,
                self.regressor_train_df,
                self.regressor_validate_df)
        else:
            raise Exception('unsupported recommender')



