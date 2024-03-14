import numpy as np
import shap

from nirdizati_light.encoding.constants import TaskGenerationType
from nirdizati_light.predictive_model.predictive_model import drop_columns


def shap_explain(CONF, predictive_model, encoder,full_test_df, target_trace_id=None,prefix_columns=None):
    test_df = drop_columns(full_test_df)

    explainer = _init_explainer(predictive_model.model, test_df)
    if CONF['predictive_model'] == 'xgboost':
        prefix_columns = [col for col in full_test_df.columns if 'prefix' in col]
    if target_trace_id is not None:
        full_test_df[full_test_df['trace_id'] == target_trace_id]
        exp = explainer(drop_columns(full_test_df))
        encoder.decode(full_test_df)
        if prefix_columns:
            full_test_df[prefix_columns] = full_test_df[prefix_columns].astype('category')
        exp.data = drop_columns(full_test_df)
        #shap.plots.waterfall(exp, show=False)
    else:
        exp = explainer(drop_columns(full_test_df))
        encoder.decode(full_test_df)
        if prefix_columns:
            full_test_df[prefix_columns] = full_test_df[prefix_columns].astype('category')
        exp.data = drop_columns(full_test_df)
        #shap.plots.bar(exp.values,show=False)
    return exp


def _init_explainer(model, df):
    try:
        return shap.TreeExplainer(model)
    except Exception as e1:
        try:
            return shap.DeepExplainer(model, df)
        except Exception as e2:
            try:
                return shap.KernelExplainer(model, df)
            except Exception as e3:
                raise Exception('model not supported by explainer')

'''
def _get_explanation(CONF, explainer, target_df, encoder):
    if CONF['task_generation_type'] == TaskGenerationType.ALL_IN_ONE.value:
        trace_ids = list(target_df['trace_id'].values)
        return {
            str(trace_id): {
                prefix_size + 1:
                    np.column_stack((
                        target_df.columns[1:-1],
                        encoder.decode_row(row)[1:-1],
                        explainer.shap_values(drop_columns(row.to_frame(0).T))[row['label'] - 1].T
                    # list(row['label'])[0]
                    )).tolist()  # is the one vs all
                for prefix_size, row in enumerate(
                    [ row for _, row in target_df[target_df['trace_id'] == trace_id].iterrows() ]
                )
                if row['label'] is not '0'
            }
            for trace_id in trace_ids
        }
    else:
        return {
            str(row['trace_id']): {
                CONF['prefix_length_strategy']:
                    np.column_stack((
                        target_df.columns[1:-1],
                        encoder.decode_row(row)[1:-1],
                        explainer.shap_values(drop_columns(row.to_frame(0).T))[row['label'] - 1].T  # list(row['label'])[0]
                    )).tolist()                                                                     # is the one vs all
            }
            for _, row in target_df.iterrows()                                                  # method!
            if row['label'] is not '0'
        }
'''
def _get_explanation(CONF, explainer, target_df, encoder):
    def process_row(row, trace_id,prefix_columns=None):
        explanation = {}
        for prefix_size, row_entry in enumerate(target_df[target_df['trace_id'] == trace_id].iterrows()):
            _, row = row_entry
            if row['label'] != '0':
                decoded_row = encoder.decode_row(row)[1:-1]
                row_df = row.to_frame(0).T
                if prefix_columns:
                    row_df = row_df[prefix_columns].astype('category')
                shap_values = explainer.shap_values(row_df)[row['label'] - 1].T
                explanation[prefix_size + 1] = np.column_stack(
                    (target_df.columns[1:-1], decoded_row, shap_values)).tolist()
                dict = {str(trace_id): explanation}

        return {str(trace_id): explanation}
    if CONF['predictive_model'] == 'xgboost':
        prefix_columns = [col for col in target_df.columns if 'prefix' in col]
    else:
        prefix_columns = None
    if CONF['task_generation_type'] == TaskGenerationType.ALL_IN_ONE.value:
        trace_ids = list(target_df['trace_id'].values)
        result = {}
        for trace_id in trace_ids:
            result.update(process_row(row, trace_id,prefix_columns))
        return result
    else:
        result = {}
        for _, row in target_df.iterrows():
            if row['label'] != '0':
                result.update(process_row(row, row['trace_id'],prefix_columns))
        return result
