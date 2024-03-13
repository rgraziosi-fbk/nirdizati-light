from enum import Enum

from nirdizati_light.explanation.wrappers.dice_wrapper import dice_explain
# from nirdizati_light.explanation.wrappers.ice_wrapper import ice_explain
from nirdizati_light.explanation.wrappers.shap_wrapper import shap_explain

class ExplainerType(Enum):
    SHAP = 'shap'
    ICE = 'ice'
    DICE = 'dice'


def explain(CONF, predictive_model, encoder, cf_df=None, test_df=None, df=None, query_instances=None, target_trace_id=None, column=None,
            method=None, optimization=None, heuristic=None, support=0.9, timestamp_col_name=None,
            model_path=None,case_ids=None,random_seed=None,case_id=None,query_instance=None,neighborhood_size=None,
            diversity_weight=None,sparsity_weight=None,proximity_weight=None,adapted=None,total_traces=None,
            minority_class=None,filtering=None,features_to_vary=None,impressed_pipeline=None,dynamic_cols=None,loreley_encoder=None,
            loreley_df=None,loreley_conf=None):
    explainer = CONF['explanator']
    if explainer is ExplainerType.SHAP.value:
        return shap_explain(CONF, predictive_model,encoder, test_df, target_trace_id=target_trace_id)
    # elif explainer is ExplainerType.ICE.value:
    #    return ice_explain(CONF, predictive_model, encoder, target_df=test_df,explanation_target=column)
    if explainer is ExplainerType.DICE.value:
        return dice_explain(CONF, predictive_model, encoder=encoder, df=df, query_instances=query_instances,
                            method=method, optimization=optimization,
                            heuristic=heuristic, support=support, timestamp_col_name=timestamp_col_name,model_path=model_path,case_ids=case_ids,
                            random_seed=random_seed,adapted=adapted,filtering=filtering,loreley_encoder=loreley_encoder,loreley_df=loreley_df,
                            loreley_conf=loreley_conf)