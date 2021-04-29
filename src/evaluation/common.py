from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score


def evaluate(actual, predicted, scores, loss=None) -> dict:
    evaluation = {}

    actual = [str(el) for el in actual]
    predicted = [str(el) for el in predicted]

    try:
        evaluation.update({'auc':roc_auc_score(actual, scores)})
    except Exception as e:
        evaluation.update({'auc': None})
    try:
        evaluation.update({'f1_score': f1_score(actual, predicted, average='macro')})
    except Exception as e:
        evaluation.update({'f1_score': None})
    try:
        evaluation.update({'accuracy': accuracy_score(actual, predicted)})
    except Exception as e:
        evaluation.update({'accuracy': None})
    try:
        evaluation.update({'precision': precision_score(actual, predicted, average='macro')})
    except Exception as e:
        evaluation.update({'precision': None})
    try:
        evaluation.update({'recall': recall_score(actual, predicted, average='macro')})
    except Exception as e:
        evaluation.update({'recall': None})

    if loss is not None:
        evaluation.update({'loss': evaluation[loss]})
    return evaluation

