import torch as t
import numpy as np


class Predictor:
    def __init__(self, test, model, batch_size, sequence_length, loss, dropout, times):
        """
        Predictor class to predict test dataset
        :param test: iterator
        :param model: trained model
        :param batch_size: batchsize
        :param sequence_length: used window, must be the same as the one used in training
        :param loss: the wanted loss function
        """
        self.test = test
        self.loss = loss
        self.model = model
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.dropout = dropout
        self.times = times

    def predict(self, ):
        """
        This function predict an entire sub-dataset from CMAPS
        :return: mean loss, true labels, predicted labels,
        True labels might be > dataset rows due to padding of the last batch while using batchsize = 1
        """
        outputs = []
        losses = []
        real = []
        stds = []

        with t.no_grad():
            for data in self.test:


                if t.cuda.is_available():

                    sequence, label = data[0].cuda(), data[1].cuda()
                else:
                    sequence, label = data[0], data[1]
                if sequence.shape[0] < self.batch_size:
                    continue

                if self.dropout or self.model.get_name() == "BAYSIAN_LSTM":
                    loss, output, y, std = self.predict_with_uncertainty(sequence, label)

                    if t.cuda.is_available():
                        std = std.cpu()
                        stds.extend(std.numpy())
                    else:
                        stds.extend(std.numpy())

                else:

                    loss, output, y = self.predict_one_sequence(sequence, label)

                losses.append(loss)
                if t.cuda.is_available():
                    output = output.cpu()
                    y = y.cpu()
                    outputs.extend(output.numpy())
                    real.extend(y.numpy())

                else:

                    outputs.extend(output.numpy())
                    real.extend(y.numpy())

            return t.mean(t.stack(losses)), np.asarray(outputs), np.asarray(real), np.asarray(stds)

    def predict_with_uncertainty(self, x, y):
        """
        Get the prediction mean and uncertainty based on standard deviation
        """

        outputs = t.zeros([self.times, y.shape[0]])
        for i in range(self.times):
            if t.cuda.is_available():
                output = self.model(x).cpu()
                y = y.cpu()
                outputs[i] = output
            else:

                outputs[i] = self.model(x)

        mean = t.mean(outputs, dim=0)
        std = t.std(outputs, dim=0)

        loss = np.sqrt(self.loss(mean, y))
        return loss, mean, y, std

    def predict_one_sequence(self, x, y):
        """
        Simple pass forward in the network
        :param x: sequence
        :param y: true label RULs
        :return:
        """
        outputs = self.model(x)

        loss = t.sqrt(self.loss(outputs, y))
        return loss, outputs, y
