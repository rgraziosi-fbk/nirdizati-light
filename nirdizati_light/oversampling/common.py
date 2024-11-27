import logging
from collections import Counter
from enum import Enum

from pandas import DataFrame, concat
from imblearn.over_sampling import SMOTENC

from nirdizati_light.encoding.data_encoder import Encoder

"""
Everything here should work also for multiclass classification
"""

class AugmentationStrategy(Enum):
    """
    Available augmentation strategies
    """
    FULL_PERC = 'full_data_percentage'
    MAJ_PERC = 'majority_class_percentage'
    MIN_PERC = 'minority_class_percentage'
    MIN_MAJ_RATIO = 'min_majority_ratio'


def get_sampling_strategy(counter_label, augmentation_strategy, augmentation_factor):

    sampling_strategy = counter_label.copy()
    majority_label = max(counter_label, key=counter_label.get)  #argmax
    majority_label_number = max(counter_label.values())
    total_sample_number = sum(counter_label.values())
    # most_minority_label = min(counter_label, key=counter_label.get)  #argmin
    total_samples_to_generate = 0
    for label, label_number in counter_label.items():
        if label != majority_label:  # only minority labels
            if augmentation_strategy == AugmentationStrategy.FULL_PERC.value:
                # number of sample to generate is a percentage of the full dataset
                samples_to_generate = round(augmentation_factor * total_sample_number)
            if augmentation_strategy == AugmentationStrategy.MAJ_PERC.value:
                # number of sample to generate is a percentage of the majority subset
                samples_to_generate = round(augmentation_factor * majority_label_number)
            if augmentation_strategy == AugmentationStrategy.MIN_PERC.value:
                # number of sample to generate is a percentage of the respective minority subset
                samples_to_generate = round(augmentation_factor * label_number)
            if augmentation_strategy == AugmentationStrategy.MIN_MAJ_RATIO.value:
                # number of sample to generate is computed so to reach (at least) a certain balancing ratio between min classes and maj class
                samples_to_generate = max(round(augmentation_factor * majority_label_number - label_number), 0)
            total_samples_to_generate += samples_to_generate
            sampling_strategy[label] += samples_to_generate

    return sampling_strategy, total_samples_to_generate


def generate_traces_smotenc(
        df: DataFrame,
        encoder: Encoder,
        augmentation_strategy: AugmentationStrategy = AugmentationStrategy.FULL_PERC.value,
        augmentation_factor: float = 0.1,
        random_state: int = 0
) -> DataFrame:
    # prepare dataset to over-sample
    print("The original train_df dataset")
    X = df.copy()
    X = X.drop(columns=['trace_id', 'label'])
    y = df['label']
    counter_label = Counter(y)
    print("Dataset before resampling:")
    print(sorted(counter_label.items()))

    # define sampling strategy
    sampling_strategy, total_samples_to_generate = get_sampling_strategy(counter_label, augmentation_strategy=augmentation_strategy, augmentation_factor=augmentation_factor)

    # instantiate SMOTENC and do resampling
    list_categorical_features = list(encoder._label_encoder.keys())
    list_categorical_features.remove('label')
    smote_nc = SMOTENC(categorical_features=list_categorical_features, random_state=random_state, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)
    print("Dataset after resampling:")
    print(sorted(Counter(y_resampled).items()))

    # build dataframe and return only the generated traces
    new_trace_id_df = DataFrame({'trace_id': [str(i+1) + '-SMOTENC' for i in range(total_samples_to_generate)]})
    new_label_df = DataFrame({'label': y_resampled.iloc[-total_samples_to_generate:].to_list()})
    new_X_df = X_resampled.iloc[-total_samples_to_generate:].copy().reset_index(drop=True)
    new_df = concat([new_trace_id_df, new_X_df, new_label_df], axis=1)

    return new_df
