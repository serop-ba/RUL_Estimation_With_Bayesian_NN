import torch

import numpy as np
from torch.utils.data import TensorDataset



class Generator:
    """
    This class to created data iterators for pytorch dataframes
    """

    def __init__(self, loader, batch_size, sequence_size):
        """
        constructor
        :param loader: parser loader containing the data frames
        :param batch_size:
        :param sequence_size:
        """
        self.loader = loader
        self.train_data = None
        self.test_data = None
        self.batch_size = batch_size
        self.sequence_size = sequence_size

    def get_train_data(self):
        """
        Get needed iterators
        supply our own training and validation samplers
        :return: training and validation iterators for training
        """
        train_tensor = torch.tensor(self.loader.train_comb[0]).float()
        label_tensor = torch.tensor(self.loader.train_comb[1]).float()
        train_data = torch.utils.data.TensorDataset(train_tensor, label_tensor)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        val_tensor = torch.tensor(self.loader.val_comb[0]).float()
        val_label_tensor = torch.tensor(self.loader.val_comb[1]).float()

        val_data = torch.utils.data.TensorDataset(val_tensor, val_label_tensor)
        testloader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=True)

        return trainloader, testloader

    def get_test_data(self):
        """
        Get test data
        :return: get 4 iterators for each sub-dataset in the test dataset
        """
        loaders = []
        test_time_data, test_labels = self.loader.test_comb
        for i in range(4):
            test_data =  np.asarray(test_time_data[i])
            test_label = test_labels[i]
            test_tensor = torch.Tensor(test_data[:,:, 2:]).float()
            label_tensor = torch.Tensor(test_label).float()
            test_data_gen = torch.utils.data.TensorDataset(test_tensor, label_tensor)
            testloader = torch.utils.data.DataLoader(test_data_gen, batch_size=self.batch_size, shuffle=False)
            loaders.append(testloader)

        return loaders
