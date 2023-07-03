import matplotlib.pyplot as plt
from numpy import max
from matplotlib import use
def bar_plot(explanations: dict, trace_id: str, prefix: int = None):
    use('TkAgg')
    plt.figure()
    trace = list(explanations.get(trace_id).values())[0]
    if prefix:
        trace = {selected_prefix: trace[selected_prefix+1] for selected_prefix in range(prefix)}
    else:
        trace = {selected_prefix: trace[selected_prefix+1] for selected_prefix in range(max(trace.keys()))}
    for event in trace:
        plt.bar_label(
            plt.barh(
                event[0]+' '+event[1],
                event[2],
                color=['red' if event[2] < 0 else 'blue']
            ),
            padding=3
        )

    plt.axvline(0, 0, 1, color="#000000", linestyle="-", linewidth=1, zorder=1)
    plt.title('Feature Importances for case ' + trace_id)
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('none')
    plt.xlim(-0.5, 0.5)
    plt.xlabel('SHAP Values')
    plt.ylabel('Features')
    plt.show()

def line_plot(explanations: dict, trace_id: str, prefix: int = None):
    use('TkAgg')
    plt.figure(figsize=(10,12),dpi=120)
    trace = list(explanations.get(trace_id).values())[0]
    '''
    if prefix:
        trace = {selected_prefix+1: trace[selected_prefix+1] for selected_prefix in range(prefix)}
    else:
        trace = {selected_prefix+1: trace[selected_prefix+1] for selected_prefix in range(max(trace.keys()))}
    '''
    importance_values = [[c for (a, b, c) in event] for event in zip(trace)]
    prefixes = [[[a, b] for (a, b, c) in event]
                for event in zip(trace)]
    prefixes = [str(i) for i in zip(prefixes)]
    plt.plot(prefixes, importance_values, marker='o',label=prefixes)
    plt.xlabel('Prefix')
    plt.ylabel('Prediction Correlation')
    plt.xticks(ticks)
    plt.ylim(-0.5, 0.5)
    plt.legend()
    plt.show()
