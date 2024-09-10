from math import sqrt
from typing import Optional

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score, mean_absolute_error, \
    mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame


def drop_columns(df: DataFrame) -> DataFrame:
    df = df.drop(['trace_id', 'label'],axis=1)
    return df

def evaluate_classifier(y_true, y_pred, scores, loss: Optional[str]=None) -> dict:
    """
    Evaluate the performance of a classifier using various metrics.

    This function calculates the AUC, F1 score, accuracy, precision, and recall of the classifier's predictions.
    It handles exceptions by setting the metric value to None if an error occurs during calculation.
    Optionally, if a loss metric is provided and exists in the evaluation, its value is updated in the evaluation dictionary.

    Args:
        y_true (1d array-like): The true labels of the data.
        y_pred (1d array-like): The predicted labels by the classifier.
        scores (1d array-like): The score/probability of the positive class.
        loss (Optional[str]): The name of the metric to be treated as a loss (the higher, the better). Defaults to None.

    Returns:
        dict: A dictionary containing the evaluation metrics: 'auc', 'f1_score', 'accuracy', 'precision', 'recall', and optionally 'loss' if specified.

    """
    evaluation = {}

    y_true = [str(el) for el in y_true]
    y_pred = [str(el) for el in y_pred]

    try:
        evaluation.update({'auc': roc_auc_score(y_true, scores)})
    except Exception as e:
        evaluation.update({'auc': None})
    try:
        evaluation.update({'f1_score': f1_score(y_true, y_pred, average='macro')})
    except Exception as e:
        evaluation.update({'f1_score': None})
    try:
        evaluation.update({'accuracy': accuracy_score(y_true, y_pred)})
    except Exception as e:
        evaluation.update({'accuracy': None})
    try:
        evaluation.update({'precision': precision_score(y_true, y_pred, average='macro')})
    except Exception as e:
        evaluation.update({'precision': None})
    try:
        evaluation.update({'recall': recall_score(y_true, y_pred, average='macro')})
    except Exception as e:
        evaluation.update({'recall': None})

    if loss is not None:    # the higher the better
        evaluation.update({'loss': evaluation[loss]})
    return evaluation


def evaluate_regressor(y_true, y_pred, loss: Optional[str]=None):
    """
    Evaluate the performance of a regression model.

    This function calculates the Root Mean Square Error (RMSE), Mean Absolute Error (MAE),
    and the R-squared score (R2 score) of the predictions made by a regression model. If a
    specific loss metric is provided and exists in the evaluation, its value will be updated
    in the evaluation dictionary.

    Args:
        y_true (1d array-like): True labels or actual values.
        y_pred (1d array-like): Predicted labels or values by the regression model.
        loss (Optional[str]): The key of the loss metric to be updated in the evaluation
                              dictionary if it exists. Defaults to None.

    Returns:
        dict: A dictionary containing the calculated metrics ('rmse', 'mae', 'rscore') with
              their corresponding values. If a metric cannot be calculated due to an exception,
              its value will be set to None. If the 'loss' argument is provided and exists in
              the evaluation, its value will be updated accordingly.
    """
    evaluation = {}

    try:
        evaluation.update({'rmse': sqrt(mean_squared_error(y_true, y_pred))})
    except Exception as e:
        evaluation.update({'rmse': None})
    try:
        evaluation.update({'mae': mean_absolute_error(y_true, y_pred)})
    except Exception as e:
        evaluation.update({'mae': None})
    try:
        evaluation.update({'rscore': r2_score(y_true, y_pred)})
    except Exception as e:
        evaluation.update({'rscore': None})
    try:
        evaluation.update({'mape': _mean_absolute_percentage_error(y_true, y_pred)})
    except Exception as e:
        evaluation.update({'mape': None})

    if loss is not None:    # the lower the better
        evaluation.update({'loss': -evaluation[loss]})
    return evaluation


def _mean_absolute_percentage_error(y_true, y_pred):
    """Calculates and returns the mean absolute percentage error

    :param y_true:
    :param y_pred:
    :return:
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if 0 in y_true:
        return -1
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate_recommender(y_true, y_pred):
    evaluation = {}

    y_true = [str(el) for el in y_true]
    y_pred = [str(el) for el in y_pred]

    try:
        evaluation.update({'f1_score': f1_score(y_true, y_pred, average='macro')})
    except Exception as e:
        evaluation.update({'f1_score': None})
    try:
        evaluation.update({'accuracy': accuracy_score(y_true, y_pred)})
    except Exception as e:
        evaluation.update({'accuracy': None})
    try:
        evaluation.update({'precision': precision_score(y_true, y_pred, average='macro')})
    except Exception as e:
        evaluation.update({'precision': None})
    try:
        evaluation.update({'recall': recall_score(y_true, y_pred, average='macro')})
    except Exception as e:
        evaluation.update({'recall': None})

    return evaluation


def evaluate_classifiers(candidates,actual):
    results = {}
    for candidate in candidates:
        predicted,scores = candidate.predict(test=True)
        result = evaluate_classifier(actual, predicted, scores)
        results[str(candidate.model_type)] = result
    return results


def plot_model_comparison(models_data: dict):
    """
    Plots a comparison of different classification models based on their performance metrics.

    This function takes a dictionary where each key is a model name and its value is another
    dictionary containing performance metrics such as F1 score, accuracy, precision, and recall.
    It then extracts these metrics and plots them for comparison.

    Args:
        models_data (dict): A dictionary where the key is the model name (str) and the value is another
                            dictionary containing metrics. The metrics dictionary should have keys
                            'f1_score', 'accuracy', 'precision', and 'recall', each mapping to a float
                            representing the model's performance on that metric.

    Returns:
        None. This function is used for plotting the comparison and does not return anything.
    """
    # Create lists to store data
    model_names = []
    f1_scores = []
    accuracies = []
    precisions = []
    recalls = []

    # Extract data from the input dictionary
    for model, metrics in models_data.items():
        model_names.append(str(model).split('(')[0])
        f1_scores.append(metrics['f1_score'])
        accuracies.append(metrics['accuracy'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])

    # Create the bubble plot
    plt.figure(figsize=(12, 8))
    plt.scatter(accuracies, precisions, s=[f1_score * 3000 for f1_score in f1_scores], c=recalls, cmap='viridis', alpha=0.7)

    # Add labels and title
    plt.title('Model Comparison')
    plt.xlabel('F1-Score')
    plt.ylabel('Precision')

    # Add color bar
    cbar = plt.colorbar()
    cbar.set_label('Recall')

    # Add text annotations for model names
    for i, txt in enumerate(model_names):
        plt.annotate(txt, (accuracies[i], precisions[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # Show plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()