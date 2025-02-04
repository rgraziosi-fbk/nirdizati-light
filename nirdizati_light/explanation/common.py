from enum import Enum

from nirdizati_light.explanation.wrappers.dice_wrapper import dice_explain
from nirdizati_light.explanation.wrappers.dice_augmentation import dice_augmentation
from nirdizati_light.explanation.wrappers.ice_wrapper import ice_explain
from nirdizati_light.explanation.wrappers.shap_wrapper import shap_explain

class ExplainerType(Enum):
    SHAP = 'shap'
    ICE = 'ice'
    DICE = 'dice'
    DICE_AUGMENTATION = 'dice_augmentation'


def explain(CONF, predictive_model, encoder, test_df=None, df=None, query_instances=None, target_trace_id=None,
            method=None, optimization=None, heuristic=None, support=0.9, timestamp_col_name=None,
            model_path=None,random_seed=None,adapted=None,filtering=None,total_traces=None,minority_class=None,
            desired_range=None,cfs_to_gen=None,features_to_vary=None):
    """
        Generate explanation based on the configuration provided in the CONF dictionary.
        :param dict CONF: dictionary for configuring the encoding
        :param nirdizati_light.predictive_model.PredictiveModel predictive_model: predictive model to explain
        :param nirdizati_light.encoding.data_encoder.Encoder encoder: encoder to use for encoding the log
        :param pandas.DataFrame test_df: test data couto evaluate model
        :param pandas.DataFrame df: full dataset
        :param pandas.DataFrame query_instances: instances to explain
        :param str target_trace_id: trace id to explain
        :param str method: method to use for explanation
        :param str optimization: optimization method to use for explanation
        :param str heuristic: heuristic to use for  counterfactual explanation
        :param float support: support for Declare model discovery for Knowledge-Aware methods
        :param str timestamp_col_name: name of the timestamp column in the log
        :param str model_path: path to save the discovered Declare model
        :param int random_seed: random seed for reproducibility
        :param bool adapted: whether to use Knowledge aware counterfactual generation method
        :param bool filtering: whether to use filtering for counterfactual explanation
        :param EventLog log: EventLog object of the log
        :param dict CONF: dictionary for configuring the encoding
        :param nirdizati_light.encoding.data_encoder.Encoder: if an encoder is provided, that encoder will be used instead of creating a new one
        :return: A list of explanations, either by providing the feature importance or the counterfactual explanations
        """
    explainer = CONF['explanator']
    if explainer is ExplainerType.SHAP.value:
        return shap_explain(CONF, predictive_model,encoder, test_df, target_trace_id=target_trace_id)
    elif explainer is ExplainerType.ICE.value:
        return ice_explain(CONF, predictive_model, encoder, target_df=test_df,explanation_target=column)
    if explainer is ExplainerType.DICE.value:
        return dice_explain(CONF, predictive_model, encoder=encoder, df=df, query_instances=query_instances,
                            method=method, optimization=optimization,
                            heuristic=heuristic, support=support, timestamp_col_name=timestamp_col_name,model_path=model_path,
                            random_seed=random_seed,adapted=adapted,filtering=filtering)
    elif explainer is ExplainerType.DICE_AUGMENTATION.value:
        return dice_augmentation(CONF=CONF, predictive_model = predictive_model, encoder = encoder, df=df, query_instances=query_instances, method = method, optimization= optimization, heuristic=heuristic, support=support,
                 timestamp_col_name=timestamp_col_name, total_traces=total_traces, minority_class=minority_class, model_path=model_path,desired_range=desired_range,cfs_to_gen=cfs_to_gen,
                                 features_to_vary=features_to_vary)