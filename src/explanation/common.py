from enum import Enum

from src.explanation.wrappers.shap_wrapper import shap_explain
from src.explanation.wrappers.lrp_wrapper import lrp_explain


class ExplainerType(Enum):
    SHAP = 'shap'
    LRP = 'lrp'


def explain(CONF, predictive_model, test_df, encoder):
    explainer = CONF['explanator']
    if explainer is ExplainerType.SHAP.value:
        return shap_explain(predictive_model, test_df, encoder)
    elif explainer is ExplainerType.LRP.value:
        return lrp_explain(CONF, predictive_model, test_df, encoder)
