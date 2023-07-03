import pm4py
import numpy as np
import pandas as pd
import os
import glob
data_dir = '/Users/andrei/Desktop/PhD/nirdizati_light/results_eval'
subfolders = os.listdir(data_dir)
print(subfolders)
path = os.path.join(data_dir,subfolders[0])
os.chdir(path)
for i ,_ in enumerate(subfolders):
    path = os.path.join(data_dir, subfolders[i])
    os.chdir(path)
    all_files = glob.glob(path + "/*simple.csv")
    for file in all_files:
        cf_csv = pd.read_csv(file)
        cf_csv.insert(loc=0, column='Case ID', value=np.divmod(np.arange(len(cf_csv)), 1)[0] + 1)
        cf_csv.insert(loc=1,column='label',value=1)
        random_df = cf_csv[cf_csv['method'] == 'random'].iloc[:,:-5]
        random_df.to_csv('100_dice_complex_random.csv')
        genetic_df = cf_csv[cf_csv['method'] == 'genetic'].iloc[:,:-5]
        genetic_df.to_csv('100_dice_complex_genetic.csv')
        kd_df = cf_csv[cf_csv['method'] == 'kdtree'].iloc[:,:-5]
        kd_df.to_csv('100_dice_complex_kdtree.csv')
        ball_df = cf_csv[cf_csv['method'] == 'balltree'].iloc[:,:-5]
        ball_df.to_csv('100_dice_complex_balltree.csv')
'''
for i ,_ in enumerate(subfolders):
    path = os.path.join(data_dir, subfolders[i])
    os.chdir(path)
    all_files = glob.glob(path + "/*simple.csv")
    for file in all_files:
        cf_csv = pd.read_csv(file)
        cf_csv.insert(loc=0, column='Case ID', value=np.divmod(np.arange(len(cf_csv)), 1)[0] + 1)
        cf_csv.insert(loc=1,column='label',value=1)
        random_df = cf_csv[cf_csv['method'] == 'random'].iloc[:,:-5]
        random_df.to_csv('100_dice_simple_random.csv')
        genetic_df = cf_csv[cf_csv['method'] == 'genetic'].iloc[:,:-5]
        genetic_df.to_csv('100_dice_simple_genetic.csv')
        kd_df = cf_csv[cf_csv['method'] == 'kdtree'].iloc[:,:-5]
        kd_df.to_csv('100_dice_simple_kdtree.csv')
        ball_df = cf_csv[cf_csv['method'] == 'balltree'].iloc[:,:-5]
        ball_df.to_csv('100_dice_simple_balltree.csv')
'''
