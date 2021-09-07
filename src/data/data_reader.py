import os
import random
from collections import deque

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

RUL_EARLY = 125


class Loader:
    def __init__(self, path, config, processed=False, standardization=None, time_series=True):
        self.path = path
        self.config = config
        self.processed = processed
        self.time_series = time_series
        self.train_comb = None
        self.test_comb = []
        self.val_comb = None
        self.train = []
        self.test = []
        self.val = []
        self.rul = []
        self.norm = standardization

    def load_processed_data(self):
        """
        Load csv files after they were saved. Choose
        :return:
        """
        print("Loading processed and combined dataframes ...\n")
        path = os.path.join(self.path, "processed")
        assert (os.path.exists(path))

        train = np.load(os.path.join(path, 'normalized_train_data.npy'))
        train_labels = np.load(os.path.join(path, 'train_labels.npy'))
        self.train_comb = (train, train_labels)

        val = np.load(os.path.join(path, 'normalized_val_data.npy'))
        val_labels = np.load(os.path.join(path, 'val_labels.npy'))
        self.val_comb = (val, val_labels)
        test_labels = []
        test_time_stamps = []
        noise = bool(int(self.config.get("DATA", "ADD_NOISE")))
        for i in range(4):
            if noise:

                test_label = np.load(os.path.join(path, "test_labels_{}_noisy.npy".format(i)))
                test = np.load(os.path.join(path, "normalized_test_data_{}_noisy.npy".format(i)))
            else:
                test_label = np.load(os.path.join(path, "test_labels_{}.npy".format(i)))
                test = np.load(os.path.join(path, "normalized_test_data_{}.npy".format(i)))
            test_time_stamps.append(test)
            test_labels.append(test_label)
            print("Found testing data Nr{} of shape {}".format(i, test.shape))
        self.test_comb = (test_time_stamps, test_labels)
        print("Found training data of shape {}".format(self.train_comb[0].shape))
        print("Found val data of shape {}".format(self.val_comb[0].shape))

    def preprocess_dataframe(self, dataframe, val, rul=None, ):
        """

        :param dataframe:
        :param val: boolean if it is validation dataset
        :param rul: for test dataset ruls are provided separately
        :return:
        """
        # TODO this function has to be adjusted for lstm
        print("Pre-Processing and combining the datasets ...\n")
        assert (len(dataframe) != 0)
        total_time_stamps = []
        total_labels = []

        prev_length = 0
        i = 0
        if rul is None and val == False:

            for df in dataframe:
                if self.norm == "MinMax":
                    self.scaler = MinMaxScaler()
                    df.iloc[:, 2:len(list(df))] = self.scaler.fit_transform(df.iloc[:, 2:len(list(df))])

                elif self.norm == "Standard":
                    self.scaler = StandardScaler()
                    df.iloc[:, 2:len(list(df))] = self.scaler.fit_transform(df.iloc[:, 2:len(list(df))])
                elif self.norm == "Robust":
                    self.scaler = RobustScaler()
                    df.iloc[:, 2:len(list(df))] = self.scaler.fit_transform(df.iloc[:, 2:len(list(df))])
                else:
                    pass

                df = self.calculate_rul_from_cycle(df)
                time_stamps, labels = self.get_time_data(df)
                total_time_stamps.extend(time_stamps.copy())
                total_labels.extend(labels.copy())
                del time_stamps, labels
                i += 1
                print("Processed Train data frame number %.f" % i)

        elif val == True:
            for df in dataframe:
                if self.norm == "MinMax":

                    df.iloc[:, 2:len(list(df))] = self.scaler.transform(df.iloc[:, 2:len(list(df))])

                elif self.norm == "Standard":

                    df.iloc[:, 2:len(list(df))] = self.scaler.transform(df.iloc[:, 2:len(list(df))])
                elif self.norm == "Robust":

                    df.iloc[:, 2:len(list(df))] = self.scaler.transform(df.iloc[:, 2:len(list(df))])
                else:
                    pass

                df = self.calculate_rul_from_cycle(df)
                time_stamps, labels = self.get_time_data(df)
                total_time_stamps.extend(time_stamps.copy())
                total_labels.extend(labels.copy())
                del time_stamps, labels
                i += 1
                print("Processed val data frame number %.f" % i)

        else:

            # TODO decide for a testing strategy after talking to An

            for df in dataframe:
                add_noise = bool(int(self.config.get("DATA", "ADD_NOISE")))
                if add_noise:
                    df = self.add_noise(df)
                if self.norm == "MinMax":
                    df.iloc[:, 2:len(list(df))] = self.scaler.transform(df.iloc[:, 2:len(list(df))])

                elif self.norm == "Standard":

                    df.iloc[:, 2:len(list(df))] = self.scaler.transform(df.iloc[:, 2:len(list(df))])
                elif self.norm == "Robust":

                    df.iloc[:, 2:len(list(df))] = self.scaler.transform(df.iloc[:, 2:len(list(df))])
                else:
                    pass

                processed_df = self.calculate_rul_from_cycle(df, np.squeeze(rul[i].values))
                time_stamps, labels = self.get_time_data(processed_df, True)
                total_time_stamps.append(time_stamps.copy())
                total_labels.append(labels.copy())
                del time_stamps, labels
                # processed_df = processed_df.drop(['cycle'], axis=1)

                # test_dataframes.append(processed_df)
                i += 1

                print("Processed Test data frame number %.f" % i)

        return (total_time_stamps, total_labels)

    def extract_rul_per_row(self, df, max_cyc=None, knee_pnts=None):
        """

        :param df:
        :param max_cyc:
        :param knee_pnts:
        :return:
        """
        RULs = []
        if max_cyc is None and knee_pnts is None:

            max_cycle = df['cycle'].max()
            knee_point = df['knee_point'].iloc[0]
        else:
            max_cycle = max_cyc
            knee_point = knee_pnts
        if knee_point < 0:
            raise ValueError("Negative knee point")
        for i in range(df.shape[0]):
            if i < knee_point:
                RULs.append(RUL_EARLY)
            else:

                rul = int(RULs[i - 1] - (RUL_EARLY / (max_cycle - knee_point)))

                if rul < 0:
                    raise ValueError("Check the formula")
                RULs.append(rul)
        return RULs

    def calculate_rul_from_cycle(self, df, max_cycles=None):
        """
        This funciton find the knee_point and calculate current rul depending on its position
        It suppose to be a step wise linear function
        :param df: dataframe
        :param max_cycles: for test data the max cycles are given seperately
        :return:
        """
        # for train, validation dataset
        if max_cycles is None:

            fd_RUL = df.groupby('engine_id')['cycle'].max().reset_index()
            fd_RUL = pd.DataFrame(fd_RUL)
            fd_RUL.columns = ['engine_id', 'max']
            fd_RUL["knee_point"] = fd_RUL['max'] - RUL_EARLY
            df = df.merge(fd_RUL, on=['engine_id'], how='left')
            ruls = []
            for i in fd_RUL["engine_id"]:
                ruls.extend(self.extract_rul_per_row(df[df['engine_id'] == i]))
            df['RUL'] = ruls
            df.drop(columns=['max', 'knee_point'], inplace=True)
        else:
            # for testing data. The maximum cycle is not equal the last entry. It is provided seperately
            ruls = []
            fd_RUL = df.groupby('engine_id')['cycle'].max().reset_index()
            fd_RUL = pd.DataFrame(fd_RUL)
            fd_RUL.columns = ['engine_id', 'max']

            df = df.merge(fd_RUL, on=['engine_id'], how='left')

            for i in range(1, df['engine_id'].max() + 1):
                maximum_cycle = df[df['engine_id'] == i]['max'].max() + max_cycles[i - 1]
                knee_pnt = maximum_cycle - RUL_EARLY

                ruls.extend(self.extract_rul_per_row(df[df['engine_id'] == i], maximum_cycle, knee_pnt))
            df['RUL'] = ruls
            df.drop(columns=['max'], inplace=True)

        return df

    def __find_files(self):
        """
        AFter downloading the data into row/CMAPSSDATA this function read each txt file
        and assign the file into train, test, rul
        :return:
        """
        print("Reading the txt files ...\n")
        train = []
        test = []
        rul = []
        path1 = os.path.join('raw', 'CMAPSSData')

        path = os.path.join(self.path, path1)
        assert (os.path.exists(path))

        for root, dirs, files in os.walk(path):
            assert (len(dirs) == 0)
            files.sort()
            for file in files:
                if file.endswith(".txt"):
                    if file.startswith("readme"):
                        continue
                    else:
                        category = file.split("_")[0]
                        if category == "RUL":
                            rul.append(os.path.join(root, file))
                        elif category == "test":
                            test.append(os.path.join(root, file))
                        elif category == "train":
                            train.append(os.path.join(root, file))
                        else:
                            raise IOError("Unknown file name in data folder")
        return train, test, rul

    def load_dataframes(self, comb_and_norm=False, save_processed=True):
        """
        Load row data and normalize it. It also combine sub-train datasets into one big train data
        :param comb_and_norm: to combine and normalize
        :param save_processed: to save the processed data into the data/processed
        :return:
        """
        print("Loading data frames from txt files ...\n")

        column_name = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                       's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                       's15', 's16', 's17', 's18', 's19', 's20', 's21']

        train, test, rul = self.__find_files()

        for path in train:
            df = pd.read_table(path, header=None, delim_whitespace=True)
            df.columns = column_name
            train, val = self.split_data(df)

            self.val.append(val)
            self.train.append(train)
        for path in test:
            df = pd.read_table(path, header=None, delim_whitespace=True)
            df.columns = column_name
            self.test.append(df.copy(deep=True))
        for path in rul:
            df = pd.read_table(path, header=None, delim_whitespace=True)
            df.columns = ["RUL"]
            self.rul.append(df.copy(deep=True))

        if comb_and_norm:
            self.train_comb = self.preprocess_dataframe(self.train, False)
            self.val_comb = self.preprocess_dataframe(self.val, True)
            self.test_comb = self.preprocess_dataframe(self.test, False, self.rul)
        if save_processed:
            self.save_df()

    def split_data(self, df):
        """
        Split data into train, validation using 20% val and 80% train, according to the engine id

        :param df: train dataframe to be splitted
        :return: train, val dataframes
        """
        num_engines = df['engine_id'].max()
        grouped = df.groupby(df.engine_id)
        val = []
        train = []

        percent = 0.2
        engines_in_val = int(num_engines * percent)

        random_numbers_val = np.sort(random.sample(range(1, num_engines), engines_in_val))

        random_numbers_train = []
        for i in range(1, num_engines + 1):
            if i not in random_numbers_val:
                random_numbers_train.append(i)

        assert (len(np.unique(random_numbers_val) == engines_in_val))
        assert (len(np.unique(random_numbers_train) == num_engines - engines_in_val))
        for i in random_numbers_val:
            val.append(grouped.get_group(i))

        for ii in np.sort(random_numbers_train):
            train.append(grouped.get_group(ii))
        return pd.concat(train).reset_index(drop=True), pd.concat(val).reset_index(drop=True)

    def save_df(self):
        """
        save processed dataframes into data/processed as csv files
        :return:
        """
        print("Saving processed dataframes ...\n")
        path = os.path.join(self.path, "processed")
        if not os.path.exists(path):
            os.mkdir(path)
            print("Created directory in " + path)
        train_time_stamps, train_labels = self.train_comb
        np.save(os.path.join(path, 'normalized_train_data.npy'), np.asarray(train_time_stamps), )
        np.save(os.path.join(path, 'train_labels.npy'), np.asarray(train_labels), )

        val_time_stamps, val_labels = self.val_comb
        np.save(os.path.join(path, 'normalized_val_data.npy'), np.asarray(val_time_stamps))
        np.save(os.path.join(path, 'val_labels.npy'), np.asarray(val_labels))

        test_time_steps, test_labels = self.test_comb
        noise = bool(int(self.config.get("DATA", "ADD_NOISE")))
        for i in range(4):
            test_time_step = test_time_steps[i]
            test_label = test_labels[i]
            if not noise:

                np.save(os.path.join(path, 'normalized_test_data_{}.npy'.format(i)), np.asarray(test_time_step))
                np.save(os.path.join(path, 'test_labels_{}.npy'.format(i)), np.asarray(test_label))
            else:
                np.save(os.path.join(path, 'normalized_test_data_{}_noisy.npy'.format(i)), np.asarray(test_time_step))
                np.save(os.path.join(path, 'test_labels_{}_noisy.npy'.format(i)), np.asarray(test_label))

            i += 1
        print("Processed dataframes are saved in " + path)

    def get_time_data(self, df, test=False):
        ids = df['engine_id'].unique()
        time_stamps = []
        labels = []
        timestep_size = 30
        aux_deque = deque(maxlen=30)
        for i in ids:
            series = df[df['engine_id'] == i]
            # starting the timestep deque
            for i in range(timestep_size):
                if test:
                    aux_deque.append(np.zeros(26))
                else:

                    aux_deque.append(np.zeros(24))

            # feed the timestamps list
            for i in range(len(series)):
                if test:
                    aux_deque.append(series.iloc[i, :-1].values)
                else:

                    aux_deque.append(series.iloc[i, 2:-1].values)
                time_stamps.append(list(aux_deque))

            # feed the labels lsit
            for i in range(len(series)):
                labels.append(series.iloc[i, -1])
            assert len(time_stamps) == len(labels), "Something went wrong"

        assert len(time_stamps) == len(labels), "Something went wrong"

        return np.asarray(time_stamps), np.asarray(labels)

    def add_noise(self, df):
        # TODO change this
        ids = [21, 24, 34, 81]

        noisy = []
        sigma = float(self.config.get("TEST", "NOISE_SIGMA"))
        mu = float(self.config.get("TEST", "NOISE_MU"))
        for i in ids:
            frame = df[df['engine_id'] == i]
            first = int(len(frame) * 0.2) - 1
            second = int(len(frame) * 0.5) - 1
            third = int(len(frame) * 0.7) - 1
            last = int(len(frame)) - 1
            frame.iloc[first, 2:] = frame.iloc[first, 2:] + np.random.normal(mu, sigma, frame.iloc[first, 2:].shape)
            frame.iloc[second, 2:] = frame.iloc[second, 2:] + np.random.normal(mu, sigma, frame.iloc[second, 2:].shape)
            frame.iloc[third, 2:] = frame.iloc[third, 2:] + np.random.normal(mu, sigma, frame.iloc[third, 2:].shape)
            frame.iloc[last, 2:] = frame.iloc[last, 2:] + np.random.normal(mu, sigma, frame.iloc[last, 2:].shape)
            df[df['engine_id'] == i] = frame

        return df
