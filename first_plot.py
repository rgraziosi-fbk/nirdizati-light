import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns

# Set the desired color palette
sns.set_palette("tab10")
datasets = ['sepsis_cases_1_start','sepsis_cases_2_start','sepsis_cases_3_start','bpic2012_2_start_old','bpic2012_2_start','bpic2015_2_start']
for dataset in datasets:
    for file in os.listdir('experiments/new_results/'):
        if dataset in file and 'mcc' in file and file.endswith('no_waiting_time_sim.csv'):
            results = pd.read_csv('experiments/new_results/' + file, sep=',')

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
    fig, axes = plt.subplots(nrows=3, ncols=len(model), figsize=(10, 10))

    # Iterate over augmentation factors
    for idx_aug,aug in enumerate(augs):
        # Create subplots for each model
        #axes = axes.flatten()

        for idx_m, m in enumerate(model):
            results_aug = results[results['Augmentation Factor'] == aug]
            results_method = results_aug[results_aug['Simulation'] == True]
            results_model = results_method[results_method['Model'] == m]

            initial = []
            for p in prefix:
                print(results_model[results_model['Prefix Length'] == p]['Initial Mcc'])
                initial.append(results_model[results_model['Prefix Length'] == p]['Initial Mcc'].mean())

            sim = []
            for p in prefix:
                print(results_model[results_model['Prefix Length'] == p]['Augmented Mcc'])
                sim.append(results_model[results_model['Prefix Length'] == p]['Augmented Mcc'].mean())

            results_method = results_aug[results_aug['Simulation'] == False]
            results_model = results_method[results_method['Model'] == 'xgboost']

            baseline = []
            for p in prefix:
                print(results_model[results_model['Prefix Length'] == p]['Augmented Mcc'])
                baseline.append(results_model[results_model['Prefix Length'] == p]['Augmented Mcc'].mean())
            print(idx_aug, idx_m)
            axes[idx_aug+idx_m].plot(initial, color='green',linewidth=3)
            axes[idx_aug+idx_m].plot(baseline, color='blue',linewidth=3)
            axes[idx_aug+idx_m].plot(sim, color='red', linewidth=3)
            axes[idx_aug+idx_m].set_xticks(range(len(prefix)))  # Set xticks based on prefix length
            axes[idx_aug+idx_m].set_xticklabels(prefix)
            axes[idx_m].set_title(model[idx_m], fontsize=20)
            axes[idx_m].set_title(model[idx_m], fontsize=20)
            if dataset == 'sepsis_cases_1_start':
                axes[idx_aug+idx_m].set_ylim(-0.2, 0.4)
        axes[idx_aug].set_ylabel(f"Augmentation Factor: {aug}", fontsize=12)
    fig.legend(['Initial', 'Baseline', 'Sim+CF'],ncols=3, loc='upper center', fontsize=20,bbox_to_anchor=(0.5, 0.97))
    fig.suptitle(dataset+'_updated',fontsize=30,y=0.995)
    #plt.tight_layout()
    plt.savefig(
        'experiments/full_plots/' + dataset + '_Mcc.png')