import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import json
import math

# Define custom activation for sigma
def custom_sigma_activation(x):
    return torch.nn.functional.elu(x) + 1

# Custom loss function
def custom_loss(gt, mu, sigma):
    sigma = torch.nn.functional.softplus(sigma)
    dist = torch.distributions.Normal(loc=mu, scale=sigma)
    log_prob = dist.log_prob(gt)
    return -log_prob.mean()  # Negative log-likelihood

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, act_res, numerical, targets):
        self.act_res = act_res
        self.numerical = numerical
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.act_res[idx], self.numerical[idx], self.targets[idx]

# Define the LSTM-based model
class CustomModel(nn.Module):
    def __init__(self, num_unique_act_res, embedding_size, window_size, feature_dim, lstm_size):
        super(CustomModel, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings=num_unique_act_res, embedding_dim=embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size + feature_dim, hidden_size=lstm_size, batch_first=True)
        self.mu_layer = nn.Linear(lstm_size, 1)
        self.sigma_layer = nn.Linear(lstm_size, 1)

    def forward(self, act_res, numerical):
        embedded = self.embedding(act_res)
        concatenated = torch.cat((embedded, numerical), dim=2)
        lstm_out, _ = self.lstm(concatenated)
        lstm_out_last = lstm_out[:, -1, :]
        
        mu = self.mu_layer(lstm_out_last)
        sigma = custom_sigma_activation(self.sigma_layer(lstm_out_last))
        return mu, sigma


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
    if NAME_EXPERIMENT == 'sepsis':
        NUMBER_2_DIAGNOSE = data["NUMBER_2_DIAGNOSE"]
        DIAGNOSE_2_NUMBER = data["DIAGNOSE_2_NUMBER"]
    PREFIX_LEN = data["PREFIX_LEN"]

PREFIX_COLUMNS = ['prefix'+str(i) for i in range(PREFIX_LEN)]
FEATURE_COLUMNS = ["caseid", "org:resource", "concept:name", "weekday", "hour"] + TRACE_ATTRIBUTES + EVENT_ATTRIBUTES + PREFIX_COLUMNS
TARGET_COLUMN = ["processing_time", "caseid"]

### create processing_time
# Load and preprocess data
df = pd.read_csv(PATH_DATA, sep=";")
df['start:timestamp'] = df['start:timestamp'].astype(str).str[:19]
df['time:timestamp'] = df['time:timestamp'].astype(str).str[:19]
df = df.sort_values(by=['caseid', 'start:timestamp'], ascending=[True, True])
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True, format="%Y-%m-%d %H:%M:%S")
df['start:timestamp'] = pd.to_datetime(df['start:timestamp'], utc=True, format="%Y-%m-%d %H:%M:%S")
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
trace_attribs = {a: [] for a in TRACE_ATTRIBUTES}
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
            index = 0
            trace_attribs[a] += [int(bool(group_case[a].iloc[index]))] * len(group_case)
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

#df.to_csv('sepsis_start_training.csv')

# Define X and y
X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

print(X.iloc[0])
print(y.iloc[0])

y_numeric = pd.to_numeric(y.iloc[:, 0], errors='coerce')
targets = torch.tensor(y.iloc[:, 0].values, dtype=torch.float32)

X_train_full, X_test, y_train_full, y_test = train_test_split(X, targets, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

train_dataset = CustomDataset(torch.tensor(X_train[:, :, 0], dtype=torch.long),
                              torch.tensor(X_train[:, :, 1:], dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))

val_dataset = CustomDataset(torch.tensor(X_val[:, :, 0], dtype=torch.long),
                            torch.tensor(X_val[:, :, 1:], dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define model and optimizer
EMBEDDING_SIZE = 4
num_unique_act_res = len(df["act_res_idx"].unique())
PREFIX_LEN = X.shape[1]  # e.g., sequence length
FEATURE_COLUMNS = X.shape[2] - 1  # exclude act_res
model = CustomModel(num_unique_act_res, EMBEDDING_SIZE, PREFIX_LEN, len(FEATURE_COLUMNS), lstm_size=10)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = custom_loss

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for act_res, numerical, targets in train_loader:
        optimizer.zero_grad()
        mu, sigma = model(act_res, numerical)
        loss = criterion(targets[:, -1:], mu, sigma)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "lstm_custom_model.pth")
print("Model saved successfully!")

########### NEW ###########
import pandas as pd
import numpy as np
import torch

# Load the model
num_unique_act_res = len(df["act_res_idx"].unique())
model = CustomModel(num_unique_act_res, EMBEDDING_SIZE, PREFIX_LEN, len(FEATURE_COLUMNS), lstm_size=10)
model.load_state_dict(torch.load("lstm_custom_model.pth"))
model.eval()

# Prepare test data
test_dataset = CustomDataset(
    torch.tensor(X_test[:, :, 0], dtype=torch.long),
    torch.tensor(X_test[:, :, 1:], dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32)
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Prediction loop
error = []
for i, (act_res, numerical, targets) in enumerate(test_loader):
    with torch.no_grad():
        mu, sigma = model(act_res, numerical)
        mu, sigma = mu.item(), sigma.item()

        # Generate prediction using normal distribution
        time_pred = np.random.normal(mu, sigma, 1)[0]

        # Decode the original sequence
        original_sequence = [NUMBER_2_RESOURCE[idx] for idx in act_res[0].tolist() if idx != 0]

        print(f"Sample {i}:")
        print(f"  Original Sequence: {original_sequence}")
        print(f"  MU={mu:.4f}, SIGMA={sigma:.4f}, PRED={time_pred:.4f}, REAL={targets[0, -1].item():.4f}")

        # Compute error
        error.append(abs(time_pred - targets[0, -1].item()))

# Report mean and standard deviation of errors
print(f"Mean Error: {np.mean(error):.4f}, Std Dev: {np.std(error):.4f}")
