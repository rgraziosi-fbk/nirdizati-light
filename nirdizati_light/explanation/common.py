from enum import Enum

from nirdizati_light.explanation.wrappers.dice_wrapper import dice_explain
from nirdizati_light.explanation.wrappers.ice_wrapper import ice_explain
from nirdizati_light.explanation.wrappers.lrp_wrapper import lrp_explain
from nirdizati_light.explanation.wrappers.shap_wrapper import shap_explain


class ExplainerType(Enum):
    SHAP = 'shap'
    LRP = 'lrp'
    ICE = 'ice'
    DICE = 'dice'


def explain(CONF, predictive_model, encoder, cf_df=None, test_df=None, df=None, query_instances=None, target_trace_id=None,
            features_to_vary=None, method=None, optimization=None, heuristic=None, support=0.99, timestamp_col_name=None):
    explainer = CONF['explanator']
    if explainer is ExplainerType.SHAP.value:
        return shap_explain(CONF, predictive_model, test_df,encoder, target_trace_id=target_trace_id)
    elif explainer is ExplainerType.LRP.value:
        return lrp_explain(CONF, predictive_model, encoder, test_df=test_df)
    elif explainer is ExplainerType.ICE.value:
        return ice_explain(CONF, predictive_model, encoder, target_df=test_df)
    elif explainer is ExplainerType.DICE.value:
        return dice_explain(CONF, predictive_model, encoder=encoder, cf_df=cf_df, df=df, query_instances=query_instances,
                            features_to_vary=features_to_vary, method=method, optimization=optimization,
                            heuristic=heuristic, support=support, timestamp_col_name=timestamp_col_name)