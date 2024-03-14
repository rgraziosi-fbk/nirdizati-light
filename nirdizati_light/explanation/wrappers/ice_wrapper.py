from pdpbox import info_plots
#from pdpbox.utils import _get_grids
from sklearn.inspection import PartialDependenceDisplay
#from src.encoding.common import get_encoded_df
#from src.encoding.models import ValueEncodings
#from src.explanation.models import Explanation


def ice_explain(CONF, predictive_model, encoder, target_df,explanation_target= None):
    if explanation_target is None:
        explanation_target = 'AMOUNT_REQ'
    else:
        explanation_target = explanation_target
    target_df = target_df.iloc[:,1:]
    feature_grids, percentile_info = _get_grids(
        feature_values=target_df[explanation_target].values, num_grid_points=10, grid_type=None,
        percentile_range='percentile', grid_range=None)
    custom_grids = [x for x in range(int(feature_grids.min()), int(feature_grids.max()))]
    fig, axes, summary_df = info_plots.target_plot(
        df = target_df,
        feature = explanation_target,
        feature_name = explanation_target,
        cust_grid_points = custom_grids,
        target = 'label',
        show_percentile = False
    )
    summary_df = summary_df[summary_df['count'] != 0]
    lists = list(target_df[explanation_target].values)
    indexs = [lists.index(x) for x in summary_df['x']]
    encoder.decode(target_df)
    values = target_df[explanation_target].values
    lst = []
    for x in range(len(indexs) - 1):
        lst.append({'value': values[indexs[x]],
                    'label': summary_df['label'].iloc[x],
                    'count': int(summary_df['count'].iloc[x]),
                    })
    return lst,fig,axes
