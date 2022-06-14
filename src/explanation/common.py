from enum import Enum

#from src.explanation.wrappers.shap_wrapper import shap_explain
#from src.explanation.wrappers.lrp_wrapper import lrp_explain
#from src.explanation.wrappers.ice_wrapper import ice_explain
from src.explanation.wrappers.dice_wrapper import dice_explain


class ExplainerType(Enum):
    SHAP = 'shap'
    LRP = 'lrp'
    ICE = 'ice'
    DICE = 'dice'


def explain(CONF, predictive_model, cf_df, encoder, df=None,query_instances=None,target_trace_id=None,features_to_vary=None,method=None):
    explainer = CONF['explanator']
    if explainer is ExplainerType.SHAP.value:
        return shap_explain(CONF, predictive_model, test_df, encoder, target_trace_id=target_trace_id)
    elif explainer is ExplainerType.LRP.value:
        return lrp_explain(CONF, predictive_model, test_df, encoder)
    elif explainer is ExplainerType.ICE.value:
        return ice_explain(CONF,predictive_model,test_df,encoder)
    else:
        return dice_explain(CONF, predictive_model, cf_df, encoder, query_instances=query_instances, features_to_vary=features_to_vary, method=method)