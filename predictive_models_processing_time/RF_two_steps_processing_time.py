import pm4py
import pandas as pd
import numpy as np
from datetime import datetime
import os
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import json
import random
import math
import warnings
warnings.filterwarnings("ignore")


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


param_grid = {
    'n_estimators': [200],
    'max_depth': [10],
    'min_samples_split': [10],
    'min_samples_leaf': [4]
}

#### PRE-PROCESSING DATA
PATH_DATA = 'sepsis_estimated_start.csv'
NAME_EXPERIMENT = 'sepsis'
PATH_SAVE_MODEL = '../datasets/' + NAME_EXPERIMENT
PATH_PARAMETERS = '../datasets/' + NAME_EXPERIMENT + '/input_' + NAME_EXPERIMENT + '.json'

#### retrieve information from json file
with open(PATH_PARAMETERS) as file:
    data = json.load(file)
    ACT_2_NUMBER = data["ACT_2_NUMBER"]
    NUMBER_2_ACT = data["NUMBER_2_ACT"]
    TRACE_ATTRIBUTES = data["TRACE_ATTRIBUTES"]
    RESOURCE_2_NUMBER = data["RESOURCE_2_NUMBER"]
    NUMBER_2_RESOURCE = data["NUMBER_2_RESOURCE"]
    EVENT_ATTRIBUTES = data["EVENT_ATTRIBUTES"]
    NUMBER_2_DIAGNOSE = data["NUMBER_2_DIAGNOSE"]
    DIAGNOSE_2_NUMBER = data["DIAGNOSE_2_NUMBER"]
    PREFIX_LEN = data["PREFIX_LEN"]

PREFIX_COLUMNS = ['prefix'+str(i) for i in range(PREFIX_LEN)]
FEATURE_COLUMNS = ["caseid", "org:resource", "concept:name", "weekday", "hour"] + TRACE_ATTRIBUTES + EVENT_ATTRIBUTES + PREFIX_COLUMNS
TARGET_COLUMN = ["processing_time", "caseid"]

### create processing_time
# Load and preprocess data
df = pd.read_csv(PATH_DATA, sep=";")
df = df.sort_values(by=['caseid', 'start:timestamp'], ascending=[True, True])
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)
df['start:timestamp'] = pd.to_datetime(df['start:timestamp'], utc=True)
df['processing_time'] = (df['time:timestamp'] - df['start:timestamp']).dt.total_seconds()
#### hour and weekday
weekday = []
hour = []
for idx, row in enumerate(df.iterrows()):
    weekday.append(row[1]['start:timestamp'].weekday())
    hour.append(row[1]['start:timestamp'].hour)
df['weekday'] = weekday
df["hour"] = hour
### traces attributes
caseid_unique = list(df['caseid'].unique())
trace_attribs = {a:[] for a in TRACE_ATTRIBUTES}
for caseid in caseid_unique:
    group_case = df[df['caseid'] == caseid]
    for a in TRACE_ATTRIBUTES:
        if a == 'Diagnose':
            if len(list(group_case[a].unique())) > 1:
                index = group_case[a].notna().to_numpy().nonzero()[0][0]
                encoding = DIAGNOSE_2_NUMBER[group_case[a].iloc[index]]
            else:
                encoding = DIAGNOSE_2_NUMBER['NOT']
            trace_attribs[a] += [encoding] * len(group_case)
        else:
            index = group_case[a].notna().to_numpy().nonzero()[0]
            trace_attribs[a] += [int(group_case[a].iloc[index])] * len(group_case)
for a in TRACE_ATTRIBUTES:
    df[a] = trace_attribs[a]

#### prefix
prefix_columns = {'prefix'+str(i): [] for i in range(PREFIX_LEN)}
event_attrib = {e: [] for e in EVENT_ATTRIBUTES}
for caseid in caseid_unique:
    group_case = df[df['caseid'] == caseid]
    prefix_case = []
    for idx, row in enumerate(group_case.iterrows()):
        prefix = 'prefix'+str(idx)
        if len(prefix_case) > PREFIX_LEN:
            prefix_case.pop(0)
        prefix_case.append(row[1]['concept:name'])

        for idx, prefix_i in enumerate(prefix_columns):
            if idx < len(prefix_case):
                prefix_columns[prefix_i].append(ACT_2_NUMBER[prefix_case[idx]])
            else:
                prefix_columns[prefix_i].append(ACT_2_NUMBER['PAD'])

        for e in EVENT_ATTRIBUTES:
            event_attrib[e].append(-1 if math.isnan(row[1][e]) else row[1][e])

### resource
df['org:resource'] = df['org:resource'].map(RESOURCE_2_NUMBER)
df['concept:name'] = df['concept:name'].map(ACT_2_NUMBER)

for e in EVENT_ATTRIBUTES:
    df[e] = event_attrib[e]

for prefix_i in prefix_columns:
    df[prefix_i] = prefix_columns[prefix_i]

df.to_csv('sepsis_start_training.csv')

# Define X and y
X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# Scale the target column y
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(df["processing_time"].values.reshape(-1, 1)).flatten()
y.loc[:, 'processing_time'] = y_scaled

unique_case_ids = df['caseid'].unique()
train_case_ids, test_case_ids = train_test_split(unique_case_ids, test_size=0.2, random_state=42)

# Create train and test datasets based on caseid
X_train = X[X['caseid'].isin(train_case_ids)]
X_test = X[X['caseid'].isin(test_case_ids)]

y_train = y[y['caseid'].isin(train_case_ids)]
y_test = y[y['caseid'].isin(test_case_ids)]

X_train = X_train.drop(['caseid'], axis=1)
X_test = X_test.drop(['caseid'], axis=1)
y_train = y_train.drop(['caseid'], axis=1)
y_test = y_test.drop(['caseid'], axis=1)

test_indices = X_test.index
case_selected = random.sample(list(test_case_ids), 20)
cases_to_optimize = df[df['caseid'].isin(case_selected)]

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

rfr_mean = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rfr_mean, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)
rfr_mean = grid_search.best_estimator_
print("Best Parameters MEAN:", grid_search.best_params_)

joblib.dump(rfr_mean, PATH_SAVE_MODEL + '/' + NAME_EXPERIMENT + '_predictive_model_mean.joblib')

mean_pred_train = rfr_mean.predict(X_train)
X_new_train = np.hstack((X_train, mean_pred_train.reshape(-1, 1)))

# Define y_new as the absolute difference between predicted and actual processing time (std)
y_new_train = np.abs(mean_pred_train - y_train)

# Train the second Random Forest Regressor (RFR) to predict the std of processing time
rfr_std = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rfr_std, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_new_train, y_new_train.ravel())
rfr_std = grid_search.best_estimator_
print("Best Parameters STD:", grid_search.best_params_)

joblib.dump(rfr_std, PATH_SAVE_MODEL + '/' + NAME_EXPERIMENT + '_predictive_model_std.joblib')
joblib.dump(scaler, PATH_SAVE_MODEL + '/' + NAME_EXPERIMENT + '_predictive_model_scaler.pkl')

mean_pred_test = rfr_mean.predict(X_test)
X_new_test = np.hstack((X_test, mean_pred_test.reshape(-1, 1)))
std_pred_test = rfr_std.predict(X_new_test)

y_pred_sampled = np.random.normal(mean_pred_test, std_pred_test)

y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_sampled_rescaled = scaler.inverse_transform(y_pred_sampled.reshape(-1, 1)).flatten()

# Calculate the Mean Absolute Error
mae_sampled = mean_absolute_error(y_test_rescaled, y_pred_sampled_rescaled)/3600

print(f"MAE: {mae_sampled}")

error_processing_time = []
error_mu = []
error_std = []

for i in range(0, round(len(X_test)/5)):
    mean_pred = rfr_mean.predict(X_test.iloc[[i]])
    X_new_test = np.hstack((X_test.iloc[[i]], mean_pred.reshape(-1, 1)))
    std_pred = rfr_std.predict(X_new_test)

    original_index = test_indices[i]  # Assuming indices were saved earlier

    proc_time_pred = np.random.normal(mean_pred, std_pred, 1)[0]
    mu_rescaled = scaler.inverse_transform(mean_pred.reshape(-1, 1))[0][0]
    sigma_rescaled = scaler.inverse_transform(std_pred.reshape(-1, 1))[0][0]
    proc_time_pred_rescaled = scaler.inverse_transform(np.array([[proc_time_pred]]))[0][0]

    #print(f"Index: {original_index}, Predicted mu: {mu_rescaled}, Predicted sigma: {sigma_rescaled}, Predicted Processing Time: {proc_time_pred_rescaled}, Real Processing Time: {df.iloc[original_index]['processing_time']}")
    #print('--------------------------------------------------------------------------------------------------------------------------')

    error_processing_time.append(abs(df.iloc[original_index]['processing_time'] - proc_time_pred_rescaled))

print('ERROR PROCESSING TIME', np.mean(error_processing_time)/3600)
