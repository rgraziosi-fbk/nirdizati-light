from collections import Counter

import pandas as pd
from numpy.random import RandomState
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTENC, SMOTE, SMOTEN


def main1():
    X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9], n_informative=3,
                               n_redundant=1, flip_y=0, n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    print(f'Original dataset shape {X.shape}')
    # Original dataset shape (1000, 20)
    print(f'Original dataset samples per class {Counter(y)}')
    # Original dataset samples per class Counter({1: 900, 0: 100})

    # simulate the 2 last columns to be categorical features
    X[:, -2:] = RandomState(10).randint(0, 4, size=(1000, 2))

    sm = SMOTENC(random_state=42, categorical_features=[18, 19])
    X_res, y_res = sm.fit_resample(X, y)
    print(f'Resampled dataset samples per class {Counter(y_res)}')
    # Resampled dataset samples per class Counter({0: 900, 1: 900})

def main2():
    X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9], n_informative=3,
                               n_redundant=1, flip_y=0, n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    print(f'Original dataset shape {X.shape}')
    # Original dataset shape (1000, 20)
    classes_counter = Counter(y)
    print(f'Original dataset samples per class {Counter(y)}')
    # Original dataset samples per class Counter({1: 900, 0: 100})

    # simulate the 2 last columns to be categorical features
    X[:, -2:] = RandomState(10).randint(0, 4, size=(1000, 2))

    new_classes_counter = classes_counter.copy()
    new_classes_counter[0] += 10
    new_classes_counter[1] += 11
    sm = SMOTENC(random_state=42, categorical_features=[18, 19], sampling_strategy=new_classes_counter)
    X_res, y_res = sm.fit_resample(X, y)
    print(f'Resampled dataset samples per class {Counter(y_res)}')
    # Resampled dataset samples per class Counter({0: 900, 1: 900})

def main3():
    data_dict = {'age': [0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
                 'class1': ['a', 'a', 'c', 'o', 'o', 'o', 'a', 'c', 'c'],
                 'class2': ['a', 'a', 'c', 'o', 'o', 'o', 'a', 'c', 'c'],
                 'class3': ['a', 'a', 'c', 'o', 'o', 'o', 'a', 'c', 'c'],
                 'class4': ['a', 'a', 'c', 'o', 'o', 'o', 'a', 'c', 'c'],
                 'class5': ['a', 'a', 'c', 'o', 'o', 'o', 'a', 'c', 'c'],
                 'class6': ['a', 'a', 'c', 'o', 'o', 'o', 'a', 'c', 'c'],
                 'class7': ['a', 'a', 'c', 'o', 'o', 'o', 'a', 'c', 'c'],
                 'class8': ['a', 'a', 'c', 'o', 'o', 'o', 'a', 'c', 'c'],
                 'class9': ['a', 'a', 'c', 'o', 'o', 'o', 'a', 'c', 'c'],
                 'class10': ['a', 'a', 'c', 'o', 'o', 'o', 'a', 'c', 'c'],
                 'class11': ['a', 'a', 'c', 'o', 'o', 'o', 'a', 'c', 'c'],
                 'class12': ['a', 'a', 'c', 'o', 'o', 'o', 'a', 'c', 'c'],
                 'class13': ['a', 'a', 'c', 'o', 'o', 'o', 'a', 'c', 'c'],
                 'class14': ['a', 'a', 'c', 'o', 'o', 'o', 'a', 'c', 'c'],
                 'class15': ['a', 'a', 'c', 'o', 'o', 'o', 'a', 'c', 'c'],
                 'class16': ['a', 'a', 'c', 'o', 'o', 'o', 'a', 'c', 'c'],
                 'class17': ['a', 'a', 'c', 'o', 'o', 'o', 'a', 'c', 'c'],
                 'class18': ['a', 'a', 'c', 'o', 'o', 'o', 'a', 'c', 'c'],
                 'class19': ['a', 'a', 'c', 'o', 'o', 'o', 'a', 'c', 'c'],
                 'class20': ['a', 'a', 'c', 'o', 'o', 'o', 'a', 'c', 'c'],
                 'label': [0, 0, 0, 1, 1, 1, 0, 0, 0]}
    df = pd.DataFrame(data_dict)
    X = df[['age', 'class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'class10',
            'class11', 'class12', 'class13', 'class14', 'class15', 'class16', 'class17', 'class18', 'class19', 'class20']]
    y = df['label']
    counter_label = Counter(y)
    print(f'Original dataset samples per class {Counter(y)}')
    majority_label = max(counter_label, key=counter_label.get)  # argmin
    majority_label_number = max(counter_label.values())
    sampling_strategy = counter_label.copy()
    sampling_strategy[majority_label] += 10
    smote = SMOTE(random_state=42, k_neighbors=2, sampling_strategy=sampling_strategy)
    smotenc = SMOTENC(random_state=42, k_neighbors=2, categorical_features=list(range(1,21)), sampling_strategy=sampling_strategy)
    smoten = SMOTEN(random_state=42, k_neighbors=2, sampling_strategy=sampling_strategy)
    # smotencn = SMOTENC(random_state=42, categorical_features=[18, 19], sampling_strategy=new_classes_counter)
    X_smote, y_smote = smote.fit_resample(X[['age']], y)
    X_smotenc, y_smotenc = smotenc.fit_resample(X, y)
    X_smoten, y_smoten = smoten.fit_resample(X, y)
    print(f'Resampled dataset samples per class SMOTE {Counter(y_smote)}')
    print(f'Resampled dataset samples per class SMOTENC {Counter(y_smotenc)}')
    print(f'Resampled dataset samples per class SMOTENC {Counter(y_smoten)}')
    print('')
    print('FINAL CONCLUSIONS:')
    print('It is clear, looking at the samples generated by SMOTENC in this experiment, that actually nominal features are used in computing the distance between samples in the continuos features space.')
    print('They are indeed weighted with some factor coming from the median of the std computed for the continuous features.')
    print('This should justify the need of both continuos and nominal features for this method.')

if __name__ == '__main__':
    main3()