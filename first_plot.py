import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns

# Set the desired color palette
sns.set_palette("tab10")

datasets = ['ConsultaDataMining201618','SynLoan','PurchasingExample','Productions']
results_combined = pd.DataFrame()  # To store data from all datasets
path = 'results_new'
# Load and concatenate results for all datasets
for dataset in datasets:
    for file in os.listdir(path):
        if dataset in file and 'mcc' in file:
            temp_results = pd.read_csv(os.path.join(path , file), sep=',')
            temp_results['Dataset'] = dataset  # Add a column to identify the dataset
            results_combined = pd.concat([results_combined, temp_results], ignore_index=True)

prefix = sorted(results_combined['Prefix Length'].unique())
model = sorted(results_combined['Model'].unique())
augs = sorted(results_combined['Augmentation Factor'].unique())
augs = [0.05,0.1,0.15]

# Set up subplots: one row for each augmentation factor, one column for each dataset
fig, axes = plt.subplots(nrows=len(augs), ncols=len(datasets), figsize=(15, 3 * len(augs)), sharey=True)

# Iterate over each augmentation factor and dataset combination
for row_idx, aug in enumerate(augs):
    for col_idx, dataset in enumerate(datasets):
        ax = axes[row_idx, col_idx]

        for m in model:
            # Filter data for the current augmentation factor, dataset, and model
            dataset_results = results_combined[
                (results_combined['Dataset'] == dataset) &
                (results_combined['Augmentation Factor'] == aug) &
                (results_combined['Model'] == m)
            ]

            # Filter for simulation data
            results_sim = dataset_results[dataset_results['Simulation'] == True]
            sim = [results_sim[results_sim['Prefix Length'] == p]['Augmented Mcc'].mean() for p in prefix]

            # Filter for baseline data (Simulation == False)
            results_baseline = dataset_results[dataset_results['Simulation'] == False]
            initial = [results_baseline[results_baseline['Prefix Length'] == p]['Initial Mcc'].mean() for p in prefix]
            baseline = [results_baseline[results_baseline['Prefix Length'] == p]['Augmented Mcc'].mean() for p in prefix]

            # Plot Initial, Baseline, and Sim+CF lines for each model
            ax.plot(prefix, initial, linestyle='-', linewidth=5, label=f'Initial', alpha=0.7)
            ax.plot(prefix, baseline, linestyle='--', linewidth=5, label=f'Baseline', alpha=0.7)
            ax.plot(prefix, sim, linestyle=':', linewidth=5, label=f'Sim+CF', alpha=0.7)

        # Set titles, labels, and legends for each subplot
        if dataset == 'ConsultaDataMining201618':
            dataset = 'ConsultaDataMining'
        ax.set_title(f"{dataset} - Aug Factor {aug}", fontsize=15)
        if row_idx == len(augs) - 1:  # Only set x-axis label on the bottom row
            ax.set_xlabel("Prefix Length", fontsize=12)
        if col_idx == 0:  # Only set y-axis label on the first column
            ax.set_ylabel("MCC", fontsize=12)
        ax.legend(loc='upper left', fontsize=10)

# Main title for the entire figure and adjust layout
fig.suptitle("MCC Comparison Across Datasets and Augmentation Factors with PRIORITY given to CFS", fontsize=18, y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.93)  # Adjust top to fit the main title

# Save and show the plot
plt.savefig('experiments/new_plots_no_priority/all_datasets_Mcc_by_aug_with_priority.png')
plt.show()
