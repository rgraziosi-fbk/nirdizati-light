import numpy as np
import shap

from src.predictive_model.predictive_model import drop_columns


def shap_explain(predictive_model, full_test_df, encoder):
    test_df = drop_columns(full_test_df)

    explainer = _init_explainer(predictive_model.model, test_df)
    importances = _get_explanation(explainer, full_test_df, encoder)

    return importances


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


def _get_explanation(explainer, target_df, encoder):
    return {
        str(row['trace_id']):
            np.column_stack((
                target_df.columns[1:-1],
                encoder.decode_row(row)[1:-1],
                explainer.shap_values(drop_columns(row.to_frame(0).T))[row['label'] - 1].T  # list(row['label'])[0]
            )).tolist()                                                                     # is the one vs all
        for _, row in target_df.iterrows()                                                  # method!
        if row['label'] is not '0'
    }

