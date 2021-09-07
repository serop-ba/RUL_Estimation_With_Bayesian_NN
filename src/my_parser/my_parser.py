from src.data.data_reader import Loader
from src.data.data_generator import Generator
from configparser import ConfigParser
from src.models.lstm import LSTM
from src.models.baysian_lstm import BayesianLstm
from src.models.train_model import Trainer
from src.models.predict_model import Predictor
from src.visualization.visualize import Visualizer
import matplotlib.pyplot as plt
import torch as t
import numpy as np
import os


class MyParser:
    """
    THis class provides a connecting point between all other class. It is the heart of this project
    """

    def __init__(self, config_path):
        """
        constructor
        :param config_path: config path
        """
        self.config = ConfigParser()
        self.config.read(config_path)
        self.output = self.config.get("OUTPUT", "OUTPUT_PATH")
        data_path = self.config.get("DATA", "PATH")
        self.pre_process = bool(int(self.config.get("DATA", "PRE_PROCESS")))
        standardization = self.config.get("DATA", "STANDARDIZATION")

        self.loader = Loader(data_path, self.config, standardization=standardization)

    def load_data(self,):
        """
        either process data or loaded pre processed data in data/processed
        set PRE_PROCESS = 0 in config.txt file to use this function
        :return:
        """
        if self.pre_process:

            self.loader.load_dataframes(True)
        else:
            self.loader.load_processed_data()

    def get_train_data(self):
        """
        Get train and validation loader from pytorch
        """
        batch_size = int(self.config.get("TRAIN", "BATCH_SIZE"))
        sequence_size = int(self.config.get("TRAIN", "SEQUENCE_WINDOW"))

        data_gen = Generator(self.loader, batch_size, sequence_size)
        train_loader, test_loader = data_gen.get_train_data()
        return train_loader, test_loader

    def load_model(self):
        """
        Load pre trained model from checkpoint path
        """
        output_path = self.config.get("TEST", "CHECKPOINT_PATH")
        checkpoint_path = os.path.join(output_path, 'checkpoint.pth')
        assert (os.path.exists(checkpoint_path))
        model = self.get_model(True)
        try:

            if t.cuda.is_available():
                model.load_state_dict(t.load(checkpoint_path, map_location=t.device('cuda')))
            else:

                model.load_state_dict(t.load(checkpoint_path, map_location=t.device('cpu')))
            if t.cuda.is_available():
                model.cuda()
        except RuntimeError:

            raise ValueError("You might be loading the wrong name from config file please change MODEL_NAME in config")

        model.eval()

        if int(self.config.get("TEST", "USE_DROPOUT")) and model.get_name() == "LSTM":
            model.drop1.train()
            model.drop2.train()
            model.drop3.train()
        return model

    def visualize_trajectory(self, dataset_number, predicted, stds, real):
        """
        visualize the trajectory of engines with the ids 21, 24, 34, 81 from each test dataset
        """
        test_datasets, _ = self.loader.test_comb
        dataset = test_datasets[dataset_number]

        dropout = int(self.config.get("TEST", "USE_DROPOUT"))
        model_name = self.config.get("MODEL", "MODEL_NAME")
        save = int(self.config.get("TEST", "SAVE_RESULTS"))

        if dropout or model_name == "BAYSIAN_LSTM":
            if len(predicted) != dataset.shape[0]:
                dataset = dataset[:len(predicted)]

            visualizer = Visualizer(self.loader)
            ids = [21, 24, 34, 81]
            for i in ids:
                positions = np.argwhere(np.max(dataset[:, :, 0], axis=1) == i)
                predictions = predicted[positions]
                std = stds[positions]
                true_label = real[positions]

                visualizer.vis_uncertainty(np.squeeze(true_label), np.squeeze(predictions), np.squeeze(std), i,
                                           dataset_number, model_name, save)

        else:

            if len(predicted) != dataset.shape[0]:
                dataset = dataset[:len(predicted)]

            visualizer = Visualizer(self.loader)
            ids = [21, 24, 34, 81]
            for i in ids:
                positions = np.argwhere(np.max(dataset[:, :, 0], axis=1) == i)
                predictions = predicted[positions]
                true_label = real[positions]

                visualizer.vis_per_id(np.squeeze(true_label), np.squeeze(predictions), i,
                                           dataset_number, save)



    def get_test_data(self):
        """
        Get test loader from pytroch
        """

        sequence_size = int(self.config.get("TRAIN", "SEQUENCE_WINDOW"))
        batch = int(self.config.get("TEST", "BATCH_SIZE"))

        data_gen = Generator(self.loader, batch, sequence_size)
        test_loaders = data_gen.get_test_data()
        return test_loaders

    def save_loss(self, train_loss, val_loss):
        """
        Save the loss figure from training
        """
        plt.plot(np.arange(len(train_loss)), train_loss, label='train loss')
        plt.plot(np.arange(len(val_loss)), val_loss, label='val loss')
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(self.output,'losses.png'))

    def get_model(self, test=False):
        """
        get the corresponding model specified in config.txt
        """
        model_name = self.config.get("MODEL", "MODEL_NAME")
        if test:
            batch_size = int(self.config.get("TEST", "BATCH_SIZE"))
            sequence_size = int(self.config.get("TRAIN", "SEQUENCE_WINDOW"))
        else:

            batch_size = int(self.config.get("TRAIN", "BATCH_SIZE"))

            sequence_size = int(self.config.get("TRAIN", "SEQUENCE_WINDOW"))
        features = int(self.config.get("DATA", "FEATURES"))
        hidden_size = 32

        if model_name == "LSTM":
            print("Loading normal LSTM")
            model = LSTM(1, features, hidden_size, sequence_size, batch_size=batch_size)
            print(model)

        elif model_name == "BAYSIAN_LSTM":
            print('Loading baysian lstm ')
            model = BayesianLstm(1, features, hidden_size, sequence_size, batch_size=batch_size)
            print(model)

        if t.cuda.is_available():
            model.cuda()
        return model

    def trjectory_generator(self, x_train, y_train, test_engine_id, sequence_length, graph_batch_size, lower_bound):
        """
        Extract training trjectories one by one
        test_engine_id = [11111111...,22222222....,...]
        """
        DEBUG = False
        num_x_sensors = x_train.shape[1]
        idx = 0
        engine_ids = test_engine_id.unique()
        if DEBUG: print("total trjectories: ", len(engine_ids))

        while True:
            for id in engine_ids:

                indexes = test_engine_id[test_engine_id == id].index
                training_data = x_train[indexes]
                if DEBUG: print("engine_id: ", id, "start", indexes[0], "end", indexes[-1], "trjectory_len:",
                                len(indexes))
                batch_size = int(training_data.shape[0] / sequence_length) + 1
                idx = indexes[0]

                x_batch = np.zeros(shape=(batch_size, sequence_length, num_x_sensors), dtype=np.float32)
                y_batch = np.zeros(shape=(batch_size, sequence_length), dtype=np.float32)

                for i in range(batch_size):

                    # Copy the sequences of data starting at this index.
                    if DEBUG: print("current idx=", idx)
                    if idx >= x_train.shape[0]:
                        if DEBUG: print("BREAK")
                        break
                    elif (idx + sequence_length) > x_train.shape[0]:
                        if DEBUG: print("BREAK", idx, x_train.shape[0], idx + sequence_length - x_train.shape[0])
                        x_tmp = x_train[idx:]
                        y_tmp = y_train[idx:]
                        remain = idx + sequence_length - x_train.shape[0]
                        x_batch[i] = np.concatenate((x_tmp, x_train[0:remain]))
                        y_batch[i] = np.concatenate((y_tmp, y_train[0:remain]))
                        break

                    x_batch[i] = x_train[idx:idx + sequence_length]

                    if idx > indexes[-1] - sequence_length:
                        y_tmp = np.copy(y_train[idx:idx + sequence_length])
                        remain = sequence_length - (
                                indexes[-1] - idx + 1)  # abs(training_data.shape[0]-sequence_length)
                        if DEBUG: print("(idx + sequence_length) > trj_len:", "remain", remain)
                        y_tmp[-remain:] = lower_bound
                        y_batch[i] = y_tmp
                    else:
                        y_batch[i] = y_train[idx:idx + sequence_length]

                    idx = idx + sequence_length

                batch_size_gap = graph_batch_size - x_batch.shape[0]
                if batch_size_gap > 0:
                    for i in range(batch_size_gap):
                        x_tmp = -0.01 * np.ones(shape=(sequence_length, num_x_sensors), dtype=np.float32)
                        y_tmp = -0.01 * np.ones(shape=(sequence_length), dtype=np.float32)
                        xx = np.append(x_batch, x_tmp)
                        x_batch = np.reshape(xx, [x_batch.shape[0] + 1, x_batch.shape[1], x_batch.shape[2]])
                        yy = np.append(y_batch, y_tmp)
                        y_batch = np.reshape(yy, [y_batch.shape[0] + 1, x_batch.shape[1]])
                yield (x_batch, y_batch)

    def get_predictor(self, test, model):
        """
        Get predictor used for testing
        """
        batch = int(self.config.get("TEST", "BATCH_SIZE"))
        sequence = int(self.config.get("TEST", "SEQUENCE_WINDOW"))
        loss_name = self.config.get("TEST", "LOSS")
        dropout = int(self.config.get("TEST", "USE_DROPOUT"))
        times = int(self.config.get("TEST", "SAMPLING_NR"))
        if loss_name == "RMSE":

            loss = t.nn.MSELoss()
        else:
            raise ValueError("LOSS FUNCTION NOT implemented yet ")
        predictor = Predictor(test, model, batch, sequence, loss, dropout, times)
        return predictor

    def get_trainer(self, train, val, model, ):
        """
        Get pytorch trainer for training
        """
        optimizer_name = self.config.get("TRAIN", "OPTIMIZER")
        batch_size = int(self.config.get("TRAIN", "BATCH_SIZE"))
        loss_name = self.config.get("TRAIN", "LOSS")
        learning_rate = float(self.config.get("TRAIN", "LEARNING_RATE"))
        epochs = int(self.config.get("TRAIN", "EPOCHS"))
        sequence_size = int(self.config.get("TRAIN", "SEQUENCE_WINDOW"))
        early_stopping = int(self.config.get("TRAIN", "EARLY_STOPPING"))
        step_size = int(self.config.get("TRAIN", "STEP_SIZE"))
        gamma = float(self.config.get("TRAIN", "GAMMA"))

        if optimizer_name == "ADAM":
            optimizer = t.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        elif optimizer_name == "SGD":
            optimizer = t.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        else:
            raise ValueError("Unknown optimizer")
        if loss_name == "RMSE":
            loss = t.nn.MSELoss()
        elif loss_name == "data_loss":
            loss = t.nn.MSELoss
        else:
            raise ValueError('unknown loss function')

        # iterations_train = self.loader.train_comb.shape[0] // (batch_size * sequence_size)
        # iterations_val = self.loader.val_comb.shape[0] // (batch_size * sequence_size)
        # iterations = (iterations_train, iterations_val)

        if t.cuda.is_available():

            trainer = Trainer(train, val, model, loss, optimizer, epochs, True, early_stopping, step_size,
                              gamma, batch_size)
        else:
            trainer = Trainer(train, val, model, loss, optimizer, epochs, False, early_stopping, step_size,
                              gamma, batch_size)

        return trainer

    def save_model(self, model):
        """
        save trained model
        """
        print("saving the model")
        output = self.config.get("OUTPUT", "OUTPUT_PATH")
        if not os.path.exists(output):
            os.mkdir(output)
            print("Create path at " + output)

        path = os.path.join(output, 'checkpoint.pth')
        t.save(model.state_dict(), path)

    def save_details(self):
        """
        writing the config.txt executed for experiments tracking
        """
        with open(os.path.join(self.output, 'config.txt'), 'w') as config_file:
            self.config.write(config_file)
