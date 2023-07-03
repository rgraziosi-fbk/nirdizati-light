import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
import seaborn as sns
import glob
plt.style.use('seaborn-paper')
path= '../results_eval_prefixes_encodings/'
all_files = glob.glob(path + "/*.csv")
cmap = cm.get_cmap('Spectral') # Colour map (there are many others)
encodings = ['simple','simple_trace','loreley','complex']
encodings = ['complex']

#datasets = ['sepsis']
#datasets = ['BPIC15']
datasets = ['BPIC15','bpic2012','sepsis']
#files = ['cfeval_synthetic_data_randomForestClassifier_dice_complex.csv','cfeval_synthetic_data_randomForestClassifier_dice_simple.csv']
def import_data(file):
    data = pd.read_csv(path+file)
    df = pd.DataFrame(data)
    return df
prefix_lengths = [5,10,15,20,25]
desired_methods = ['random','kdtree','genetic','genetic_conformance']
desired_methods_declare = ['random','kdtree','genetic','original_data']
def create_plot(df,encoding=None,plot_name=None):
    df['hit_rate'] = df['generated_cfs']/df['desired_nr_of_cfs']
    df = df[df['method'].isin(desired_methods)]

    #df_groupby = df.groupby(['encoding', 'desired_nr_of_cfs']).mean()
#    df_groupby = df.groupby(['encoding', 'prefix_length']).mean()

    #df_groupby = df.groupby(['method', 'desired_nr_of_cfs']).mean()
    df_groupby = df.groupby(['method', 'prefix_length']).mean()
    name = df.iloc[0, 0]
    # crate the figure and axes
    df_groupby = df_groupby[
        ['distance_l2j', 'implausibility_nbr_cf',
         'diversity_l2j','hit_rate','runtime','avg_nbr_changes_per_cf']]
    '''
    declare = pd.read_csv('/Users/andrei/Desktop/PhD/declare4py/declare4py/tutorials/i_swear_i_am_done.csv',
                          index_col=[0], sep=';')
    declare = declare[declare['method'].isin(desired_methods_declare)]
    declare_groupby = declare.groupby(['method', 'prefix_length']).mean()
    #df_groupby['hit_rate'] = declare_groupby['hit_rate']
    declare['encoding'] = declare['encoding'].replace(['simptrace'],'simple_trace')
    if encoding is not None:
        #declare = declare[(declare['encoding'] == encoding) & (declare['dataset'].str.contains(plot_name))]
        #declare = declare[(declare['encoding'] == encoding)]
        #declare = declare.groupby(['method', 'prefix_length']).mean()
        #declare = declare.groupby(['method', 'desired_cfs']).mean()
        df_groupby['declare_sat_score'] = declare['sat_score']
    else:
#        declare = declare[declare['dataset'].str.contains(plot_name)]
        declare = declare.groupby(['method', 'prefix_length']).mean()
        #declare = declare.groupby(['method', 'desired_cfs']).mean()



        #declare = declare.groupby(['encoding','prefix_length']).mean()
        #declare = declare.groupby(['encoding', 'desired_cfs']).mean()

        #df_groupby['declare_sat_score'] = declare['sat_score_no_data']
        declare = declare['sat_score']
    '''
    # unpack all the axes subplots
    fig, axes = plt.subplots(nrows=3, ncols=7, sharex=True, figsize=(16, 4))
    # df_groupby[[j for j in df_groupby.columns]].unstack(level=0).plot(subplots=True,#axes=axes)
    for j, i in enumerate(df_groupby.columns):
        df_groupby[[i]].unstack(level=0).plot(ax=axes.flat[j], legend=None, title=i,linewidth=3.0,
                                              #style=[':', '-', '--',':']
                                               fontsize=14,color=['red', 'orange', 'green', 'blue']
                                              )
    #declare.unstack(level=0).plot(ax=axes[-1], legend=None,
    #                              title='declare_sat_score', linewidth=4.0,
    #                              # style=[':', '-', '--']
                                  # cmap="Accent",
    #                              fontsize=14, color=['red', 'orange', 'blue', 'green'])
   # fig.legend(['genetic', 'kdtree','random'],loc='upper center',
    #           ncol=4, fancybox=True, shadow=True, prop={"size": 25})

    handles, labels = axes.flat[-1].get_legend_handles_labels()
    #plt.xlim([5, 25])
    fig.legend(handles, labels, loc="upper left",bbox_to_anchor=(0.5, 0.99),
                ncol=4,prop={"size":12})
    #fig.legend(['complex', 'loreley', 'simple','simple_trace'],loc='upper center',
    #            bbox_to_anchor=(0.5, 0.95),
    #            ncol=5,prop={"size":14})
    #plt.xticks([5,10,15,20],fontsize=16)
    plt.xticks([5,10,15,20,25],fontsize=16)
    if encoding is not None:
        plt.suptitle(plot_name,fontsize=15,y=0.6)
        plt.tight_layout(rect=[0, 0, 1, 0.85])
        df_groupby.to_csv('../new_plots_21_11/%s_%s.csv' % (plot_name,encoding))
        #plt.savefig('../check_new_plots_vac_true/cf_evaluation_fig_%s_%s_pref_length' % (plot_name,encoding))
        plt.savefig('../check_new_plots_vac_true/cf_evaluation_fig_%s_%s_nr_of_cfs' % (plot_name,encoding))

    else:
        plt.suptitle(plot_name,fontsize=15,y=0.8)
        plt.tight_layout(rect=[0,0,1,0.85])
        df_groupby.to_csv('../new_plots_21_11/%s_desired_cfs.csv' % (plot_name))
        #plt.savefig('../check_new_plots_vac_true/cf_eval_%s_nr_of_cfs' % (plot_name))
        plt.savefig('../check_new_plots_vac_true/cf_eval_%s_pref_length' % (plot_name))
'''
for encoding in encodings:
    dfs = []
    for dataset in datasets:
        for file in all_files:
            if encoding in file:
                    df = pd.read_csv(file)
                    df['encoding']=encoding
                    dfs.append(df)
    dataframes = pd.concat(dfs)
    create_plot(dataframes,encoding,plot_name=encoding)
'''
'''
for dataset in datasets:
    dfs = []
    for encoding in encodings:
        for file in all_files:
            if dataset in file:
                if encoding in file:
                    print(file)
                    df = pd.read_csv(file)
                    df['encoding'] =encoding
                    dfs.append(df)
dataframes = pd.concat(dfs)
create_plot(dataframes,plot_name='Average results for each encoding')
'''
'''
for dataset in datasets:
    dfs = []
    for file in all_files:
        if dataset in file:
                df = pd.read_csv(file)
                dfs.append(df)
aggregate_data = pd.concat(dfs)
create_plot(aggregate_data,plot_name="Overall average")
'''

def create_plot_new(all_data,plot_name=None,dataset=None):
    all_data['hit_rate'] = all_data['generated_cfs']/all_data['desired_nr_of_cfs']
    all_data = all_data[all_data['method'].isin(desired_methods)]
    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(26, 21),sharex='all',sharey='col')
    rows = 0
    for dataset in datasets:

        df = all_data[all_data['dataset'].str.contains(dataset)]
        #df = all_data[all_data['encoding']==encoding]
        #df_groupby = df.groupby(['method', 'desired_nr_of_cfs']).mean()
        #df_groupby = df.groupby(['method', 'prefix_length']).mean()


        #df_groupby = df.groupby(['method', 'desired_nr_of_cfs']).mean()
        df_groupby = df.groupby(['method', 'prefix_length']).mean()
        # crate the figure and axes
        df_groupby = df_groupby[
            ['distance_l2j', 'implausibility_nbr_cf',
             'diversity_l2j','avg_nbr_changes_per_cf','hit_rate','runtime',]]

        declare = pd.read_csv('/Users/andrei/Desktop/PhD/declare4py/declare4py/tutorials/i_swear_i_am_done.csv',
                          index_col=[0],sep=';')
        declare = declare[declare['method'].isin(desired_methods_declare)]
        declare['encoding'] = declare['encoding'].replace(['simptrace'], 'simple_trace')

        if dataset is not None:
            declare = declare[(declare['dataset'].str.contains(dataset))]
        #declare = declare[(declare['encoding'] == encoding)]
        #declare = declare[(declare['dataset'].str.contains(dataset))]
        declare = declare.groupby(['method', 'prefix_length']).mean()
        if encoding == 'loreley':
            df_groupby['hit_rate'] = declare['hit_rate']
        df_groupby['hit_rate'].where(df_groupby['hit_rate'] <= 1, 1, inplace=True)
        declare = declare['sat_score']
        #declare = declare.groupby(['method', 'desired_cfs']).mean()
        #df_groupby['declare_sat_score'] = declare['sat_score']
        #df_groupby['hit_rate'].where(df['hit_rate'] <= 1, 1, inplace=True)
        for j, i in enumerate(df_groupby.columns):
            df_groupby[[i]].unstack(level=0).plot(ax=axes[rows,j], legend=None, title=i,linewidth=4.0,
                                                  #style=[':', '-', '--']
                                                  #cmap="Accent",
                                                  fontsize=18,color=['red','orange','green','blue']
                                                  )
        #declare.unstack(level=0).plot(ax=axes[rows,6],legend=None,
        #                              title='declare_sat_score',linewidth=4.0,
        #                                          #style=[':', '-', '--']
        #                                          #cmap="Accent",
        #                                          fontsize=18,color=['red','orange','blue','green'])
   #         plt.xticks([], [])
            #axes[rows, j].tick_params(axis='y',which='major')
            #axes[rows,j].set_visible(True)
        for ax in axes.flatten():
            ax.yaxis.set_tick_params(labelleft=True,labelsize=18)
            ax.set_title(label=ax.title._text,fontsize=18,fontweight='bold')
        if dataset == 'bpic2012':
            dataset = 'BPIC2012'
        elif dataset == 'sepsis':
            dataset = 'SEPSIS'
        axes[rows,3].set_title(dataset+"\n"+'avg_nbr_of_changes',fontweight="bold",size=16)
        rows+=1
    #for ax in axes.flatten():
     #   for tk in ax.get_yticklabels():
     #       tk.set_visible(True)
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    #fig.legend(handles, labels, loc="upper left",bbox_to_anchor=(0.65, 0.88),
    #            ncol=4,prop={"size":18})
    fig.legend([ 'genetic','genetic_conformance', 'kdtree','random'],loc='upper center',
                bbox_to_anchor=(0.8, 0.99),
                ncol=4,prop={"size":17})
    #plt.xticks([5,10,15,20],fontsize=16)
    #plt.xticks([5,10,15,20,25],fontsize=16)

    '''
    if encoding is not None:
        plt.suptitle(plot_name,fontsize=15,y=0.85)
        plt.tight_layout(rect=[0, 0, 1, 0.85])
        df_groupby.to_csv('../new_plots_21_11/%s_%s.csv' % (plot_name,encoding))
        #plt.savefig('../check_new_plots_vac_true/multiple_eval_%s_pref_length' % (plot_name))
        plt.savefig('../check_new_plots_vac_true/multiple_eval_fig_%s_nr_of_cfs' % (plot_name))
    '''
#    else:
    #plt.suptitle(plot_name,fontsize=15,y=0.85)
    plt.tight_layout(rect=[0,0,1,0.85])
    plt.suptitle(plot_name, fontsize=22, y=0.87,fontweight='bold')

    #plt.savefig('../test_plots/multiple_eval_%s_nr_of_cfs' % (plot_name))
    plt.savefig('../test_plots/cf_evaluation_fig_%s_pref_length_vac_long' % (plot_name),bbox_inches="tight")

dfs = []

for file in all_files:
    for encoding in encodings:
        if encoding in file:
            print(file)
            df = pd.read_csv(file)
            df['encoding']=encoding
            dfs.append(df)
dataframes = pd.concat(dfs)
create_plot_new(dataframes,plot_name='Average results for datasets')
