import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
results = pd.read_csv('experiments/model_performances_mcc_sepsis.csv', sep= ';')

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
        initial.append(float(results_model[results_model['Prefix Length'] == p]['Initial MCC']))
    print('INITIAL', initial)
    sim = []
    for p in prefix:
        sim.append(float(results_model[results_model['Prefix Length'] == p]['Augmented MCC']))
    print('SIM+CF', sim)
    ### baseline
    results_method = results_aug[results_aug['Simulation'] == False]
    results_model = results_method[results_method['Model'] == 'xgboost']
    baseline = []
    for p in prefix:
        baseline.append(float(results_model[results_model['Prefix Length'] == p]['Augmented MCC']))
    print('BASELINE', baseline)
    plt.title(m, fontsize=20)
    plt.plot(initial, color='green')
    plt.plot(baseline, color='blue')
    plt.plot(sim, color='red')
    plt.legend(['Initial', 'Baseline', 'Sim+CF'])
    plt.show()
'''
#augmentation_factors = [0.3, 0.5, 0.7]
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))

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
            print(results_model[results_model['Prefix Length'] == p]['Initial MCC'])
            initial.append(results_model[results_model['Prefix Length'] == p]['Initial MCC'].mean())

        sim = []
        for p in prefix:
            print(results_model[results_model['Prefix Length'] == p]['Augmented MCC'])
            sim.append(results_model[results_model['Prefix Length'] == p]['Augmented MCC'].mean())

        results_method = results_aug[results_aug['Simulation'] == False]
        results_model = results_method[results_method['Model'] == 'xgboost']

        baseline = []
        for p in prefix:
            print(results_model[results_model['Prefix Length'] == p]['Augmented MCC'])
            baseline.append(results_model[results_model['Prefix Length'] == p]['Augmented MCC'].mean())
        print(idx_aug, idx_m)
        axes[idx_aug][idx_m].plot(initial, color='green')
        axes[idx_aug][idx_m].plot(baseline, color='blue')
        axes[idx_aug][idx_m].plot(sim, color='red')
        axes[idx_aug][idx_m].set_title(model[idx_m], fontsize=20)
        axes[idx_aug][idx_m].set_title(model[idx_m], fontsize=20)
fig.legend(['Initial', 'Baseline', 'Sim+CF'])

plt.tight_layout()
plt.show()