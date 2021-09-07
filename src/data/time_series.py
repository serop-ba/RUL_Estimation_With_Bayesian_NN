from torch.utils.data import Dataset
import torch
import numpy as np

from torch.utils.data.sampler import Sampler


class TimeSeriesDataset(Dataset):
    """
    This class will return datasamples and true labels from one dataframe
    """

    def __init__(self, data, sequence_length, mode):
        self.sequence_length = sequence_length
        self.end = 0
        self.mode = mode
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        """

        :param item: [sequence_length, feature_length]
        :return: data and true labels separated
        """
        x_data = item[:, :-1]
        y_data = item[:, -1:]

        return [torch.Tensor(x_data), torch.Tensor(y_data)]


class TimeSeriesSampler(Sampler):
    """
    Time series sampler. We need data with the shape [batchsize, sequence_length, feature_length]
    This provides the functionality of iterating through the data and returning the wanted shape
    """

    def __init__(self, data, sequence_length, batch_size, mode):
        self.num_samples = data.shape[0]
        self.idx = 0
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.max_sample = (data.shape[0]) // sequence_length
        self.data = data
        self.mode = mode
        self.feature_length = data.shape[1]

    def __iter__(self):
        """

        :return: either train or test sampler
        """
        if self.mode == 'train':
            return self.__sample_train()
        elif self.mode == "test":
            return self.__sample_test()

    def __sample_test(self):
        """
        for batch == 1 return all samples and padd missing values
        for batch > 1 return only fully occupied batches( cropped)

        :return:
        """

        data = []

        for i in range(self.batch_size):
            if self.idx + self.sequence_length > self.num_samples:
                # TODO add padding for batch one
                if self.batch_size == 1:

                    left = self.num_samples - self.idx - 1
                    if left < 0:
                        # return an empty list
                        data = []
                        print("exceeded data samples")
                        break

                    zeros = np.zeros([self.sequence_length, self.feature_length - 1])
                    zeros[0:left] = self.data[self.idx:self.idx + left, 1:]
                    data.append(zeros)
                else:
                    # discard the entire batch
                    data = []
                    break

            temp3 = self.data[self.idx:self.idx + self.sequence_length, 1:]

            data.append(temp3)
            self.idx = self.idx + self.sequence_length
        return iter(data)

    def __sample_train(self):
        """
        This is the sampler for training and validation datasets
        It tries to fill a whole batch with sequences of shape [sequence_length, feature_size]
        When the dataset is almost empty the iterator starts from the beggining
        :return: iteratable samples of desired size
        """
        data = []

        for i in range(self.batch_size):
            if self.idx + self.sequence_length > self.num_samples:

                if self.idx == self.num_samples:
                    self.idx = 0
                    temp = self.data[self.idx:self.idx + self.sequence_length]
                    assert (len(temp) == self.sequence_length)
                    data.append(np.asarray(temp))
                    continue
                temp = []
                diff = self.num_samples - self.idx - 1

                left = self.sequence_length - diff
                temp.extend(self.data[self.idx:self.idx + diff])
                self.idx = 0
                temp.extend(self.data[self.idx:left])
                self.idx += left
                if len(temp) != self.sequence_length:
                    print('Temp size is not equal')
                    print(self.idx, diff, left, self.num_samples, )
                data.append(np.asarray(temp))
                continue

            temp3 = self.data[self.idx:self.idx + self.sequence_length]

            data.append(temp3)
            self.idx = self.idx + self.sequence_length
        return iter(data)

    def __len__(self):
        """
        important!! when called to the iterator it returns self.num_samples/ batch_size

        :return: back size of the iterator
        """
        return self.num_samples
