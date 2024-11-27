from torch import nn

from abc import ABC, abstractmethod

class CustomPytorchModel(nn.Module, ABC):
    @abstractmethod
    def __init__(self, input_dim, output_dim, config):
        # input_dim and output_dim will be computed by PredictiveModel based on the data
        # config is the hyperparameter optimization space; you can provide a custom one using the hyperopt_space attribute on PredictiveModel class
        super().__init__()

    @abstractmethod
    def forward(self, x):
        # x is a tensor of shape (num_examples, trace_length, feature_size)
        pass

class CustomModelExample(CustomPytorchModel):
    def __init__(self, input_dim, output_dim, config):
        super(CustomModelExample, self).__init__(input_dim, output_dim, config)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=int(config['lstm_hidden_size']),
            num_layers=int(config['lstm_num_layers']),
            batch_first=True
        )
        self.linear1 = nn.Linear(int(config['lstm_hidden_size']), int(config['lstm_hidden_size']) // 2)
        self.linear2 = nn.Linear(int(config['lstm_hidden_size']) // 2, output_dim)
        self.softmax = nn.Softmax(dim=1)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x