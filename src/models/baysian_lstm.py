from torch.nn import Module
import torch.nn as nn
import torch
import torch.nn.functional as F
from blitz.modules import BayesianLSTM, BayesianLinear
from blitz.utils import variational_estimator


@variational_estimator
class BayesianLstm(Module):
    def __init__(self, output_size=1, input_size=24, hidden_size=32, seq_length=30, hidden_neurons=8, batch_size=512):
        super(BayesianLstm, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.hidden_layer_size = hidden_size
        if torch.cuda.is_available():

            self.hidden_cell_1 = (torch.zeros(1, batch_size, self.hidden_layer_size).to('cuda'),
                                  torch.zeros(1, batch_size, self.hidden_layer_size).to('cuda'))
            self.hidden_cell_2 = (torch.zeros(1, batch_size, self.hidden_layer_size * 2).to('cuda'),
                                  torch.zeros(1, batch_size, self.hidden_layer_size * 2).to('cuda'))
        else:
            self.hidden_cell_1 = (torch.zeros(1, batch_size, self.hidden_layer_size).to('cpu'),
                                  torch.zeros(1, batch_size, self.hidden_layer_size).to('cpu'))
            self.hidden_cell_2 = (torch.zeros(1, batch_size, self.hidden_layer_size * 2).to('cpu'),
                                  torch.zeros(1, batch_size, self.hidden_layer_size * 2).to('cpu'))

        # First lstm cell
        self.lstm1 = BayesianLSTM(input_size, hidden_size)

        # second lstm cell
        self.lstm2 = BayesianLSTM(hidden_size, hidden_size * 2)

        # first fully connected layer
        self.fc1 = BayesianLinear(hidden_size * 2, hidden_neurons)
        self.act1 = nn.ReLU()
        # self.bat1 = nn.BatchNorm1d(num_features=hidden_neurons)

        # second fully connected layer
        self.fc2 = BayesianLinear(hidden_neurons, hidden_neurons)
        self.act2 = nn.ReLU()
        # self.bat2 = nn.BatchNorm1d(num_features=hidden_neurons)

        # output
        self.output = BayesianLinear(hidden_neurons, output_size)

    def forward(self, input_seq):
        """
        Forward pass through the network
        :param input_seq: the sequence of sensor measurements
        :return: predicted array flatted where each value is the RUL prediction for each row of sensor measurments
        """
        lstm_out = self.lstm1(input_seq)[0]
        lstm_out2 = self.lstm2(lstm_out)[0]
        out = lstm_out2[:, -1, :]

        fc1 = self.act1(self.fc1(out))
        fc2 = self.act2(self.fc2(fc1))
        output = self.output(fc2)
        return output.view(-1)

    def get_name(self):
        return 'BAYSIAN_LSTM'
