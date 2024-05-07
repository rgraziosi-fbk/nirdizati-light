import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
results = pd.read_csv('/Users/francescameneghello/Documents/GitHub/nirdizati-light/experiments/model_performances_mcc_sepsis.csv', sep= ';')

prefix = list(results['Prefix Length'].unique())
model = list(results['Model'].unique())
aug = list(results['Augmentation Factor'].unique())

#####initial and sim
for m in model:
    results_aug = results[results['Augmentation Factor'] == 0.3]
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
    plt.plot(initial, color='green')
    plt.plot(baseline, color='blue')
    plt.plot(sim, color='red')
    plt.show()