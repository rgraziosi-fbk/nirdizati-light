import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns

# Set the desired color palette
sns.set_palette("tab10")
datasets = ['sepsis_cases_1_start', 'sepsis_cases_2_start', 'sepsis_cases_3_start']
for dataset in datasets:
    for file in os.listdir('experiments/'):
        if dataset in file and 'mae' in file and file.endswith('.csv'):
            results = pd.read_csv('experiments/' + file, sep=',')
#    results = pd.read_csv('experiments/model_performances_Mcc_sepsis.csv', sep= ';')

    prefix = list(results['Prefix Length'].unique())
    model = list(results['Model'].unique())
    augs = list(results['Augmentation Factor'].unique())
    '''
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(45, 35))
    #####initial and sim
    for m in model:
        results_aug = results[results['Augmentation Factor'] == aug]
        results_method = results_aug[results_aug['Simulation'] == True]
        results_model = results_method[results_method['Model'] == m]
        initial = []
        for p in prefix:
            initial.append(float(results_model[results_model['Prefix Length'] == p]['Initial Mcc']))
        print('INITIAL', initial)
        sim = []
        for p in prefix:
            sim.append(float(results_model[results_model['Prefix Length'] == p]['Augmented Mcc']))
        print('SIM+CF', sim)
        ### baseline
        results_method = results_aug[results_aug['Simulation'] == False]
        results_model = results_method[results_method['Model'] == 'xgboost']
        baseline = []
        for p in prefix:
            baseline.append(float(results_model[results_model['Prefix Length'] == p]['Augmented Mcc']))
        print('BASELINE', baseline)
        plt.title(m, fontsize=20)
        plt.plot(initial, color='green')
        plt.plot(baseline, color='blue')
        plt.plot(sim, color='red')
        plt.legend(['Initial', 'Baseline', 'Sim+CF'])
        plt.show()
    '''
    #augmentation_factors = [0.3, 0.5, 0.7]
    initial_metrics = ['Initial Rmse', 'Initial Mae', 'Initial Rscore', 'Initial Mape', 'Initial Loss']
    augmented_metrics = ['Augmented Rmse', 'Augmented Mae', 'Augmented Rscore', 'Augmented Mape', 'Augmented Loss']

    fig, axes = plt.subplots(nrows=3, ncols=len(initial_metrics), figsize=(10, 5))

    # Iterate over augmentation factors
    for idx_metric, (initial_metric, augmented_metric) in enumerate(zip(initial_metrics, augmented_metrics)):
        for idx_aug, aug in enumerate(augs):
            for idx_m, m in enumerate(model):
                results_aug = results[results['Augmentation Factor'] == aug]
                results_method = results_aug[results_aug['Simulation'] == True]
                results_model = results_method[results_method['Model'] == m]

                initial = []
                for p in prefix:
                    print(results_model[results_model['Prefix Length'] == p][initial_metric])
                    initial.append(results_model[results_model['Prefix Length'] == p][initial_metric].mean())

                sim = []
                for p in prefix:
                    print(results_model[results_model['Prefix Length'] == p][augmented_metric])
                    sim.append(results_model[results_model['Prefix Length'] == p][augmented_metric].mean())

                results_baseline = results_aug[results_aug['Simulation'] == False]
               # results_model = results_method[results_method['Model'] == 'xgboost']

                baseline = []
                for p in prefix:
                    print(results_baseline[results_baseline['Prefix Length'] == p][augmented_metric])
                    baseline.append(results_baseline[results_baseline['Prefix Length'] == p][augmented_metric].mean())
                axes[idx_aug][idx_metric].plot(initial, color='green', linewidth=3)
                axes[idx_aug][idx_metric].plot(baseline, color='blue', linewidth=3)
                axes[idx_aug][idx_metric].plot(sim, color='red', linewidth=3)
                axes[idx_aug][idx_metric].set_title(f"{initial_metric.split(' ')[1]}", fontsize=12)
            axes[idx_aug][0].set_ylabel(f"Augmentation Factor: {aug}", fontsize=12)
    fig.legend(['Initial', 'Baseline', 'Sim+CF'],ncols=3, loc='upper center', fontsize=20,bbox_to_anchor=(0.5, 0.97))
    fig.suptitle(dataset,fontsize=30,y=0.995)
    #plt.tight_layout()
    plt.savefig(
        'experiments/plots/' + dataset + '_regression.png')