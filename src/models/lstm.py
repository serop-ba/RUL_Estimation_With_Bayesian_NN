from torch.nn import Module
import torch.nn as nn
import torch

class LSTM(Module):
    """
    This is a general 2 layer lstm with 2 fully connected layers
    L(32:63, 8:8) is the number of neurons in  each layer
    """

    def __init__(self, output_size=1, input_size=24, hidden_size=32, seq_length=30, hidden_neurons=8, batch_size=512):
        super(LSTM, self).__init__()
        self.output_size = output_size

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
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                             num_layers=1, batch_first=True, )
        self.drop1 = nn.Dropout(p=0.5)
        # second lstm cell
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size * 2,
                             num_layers=1, batch_first=True)
        self.drop2 = nn.Dropout(p=0.5)
        # first fully connected layer
        self.fc1 = nn.Linear(hidden_size * 2, hidden_neurons)
        self.act1 = nn.ReLU()
        self.bat1 = nn.BatchNorm1d(num_features=hidden_neurons)
        self.drop3 = nn.Dropout(p=0.5)

        # second fully connected layer
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.act2 = nn.ReLU()
        self.bat2 = nn.BatchNorm1d(num_features=hidden_neurons)

        # output
        self.output = nn.Linear(hidden_neurons, output_size)

    def forward(self, input_seq):
        """
        Forward pass through the network
        :param input_seq: the sequence of sensor measurements
        :return: predicted array flatted where each value is the RUL prediction for each row of sensor measurments
        """
        lstm_out= self.drop1(self.lstm1(input_seq, self.hidden_cell_1)[0])
        lstm_out2= self.drop2(self.lstm2(lstm_out, self.hidden_cell_2)[0])
        # out = lstm_out2.contiguous().view(-1, self.hidden_layer_size * 2)
        out = lstm_out2[:, -1, :]
        fc1 = self.drop3(self.bat1(self.act1(self.fc1(out))))
        fc2 = self.bat2(self.act2(self.fc2(fc1)))
        output = self.output(fc2)
        return output.view(-1)

    def get_name(self):
        return 'LSTM'
