import os
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    """
    This class will handle all visualization needed
    """

    def __init__(self, data_loader):
        self.data_loader = data_loader

    def viz_corr(self):
        """
        visualize the correlation between all features of the train datasets combined
        :return:
        """
        correlation = self.data_loader.train_comb.corr()
        sns.heatmap(correlation, cmap='RdYlGn', annot=True)
        fig = plt.gcf()
        fig.set_size_inches(20, 20)
        plt.show()

    def viz_pairplot(self):
        """
         pairs of features of the train dataset
        :return:
        """
        sns.pairplot(self.data_loader.train_comb, corner=True)
        plt.show()

    def vis_feature_hist(self, name):
        """
        plot histogram of a specific feature
        :param name: feature name
        :return:
        """
        sns.distplot(self.data_loader.train_comb[name])
        plt.show()

    def vis_feature_line(self, name):
        """

        line plot a specific feature
        :param name: name of the feature
        :return:
        """
        sns.lineplot(self.data_loader.train_comb[name])
        plt.show()

    def vis_per_id(self, true_label, predicted, i, dataset_number, save):
        """
        Show a figure depending on the engine id RUL prediction
        """

        cycles = [x for x in range(len(true_label))]
        assert (len(true_label) == len(predicted))
        fig, ax = plt.subplots(figsize=(10, 6))

        # Same as above
        ax.set_xlabel('Cycles')
        ax.set_ylabel("RUL")
        title_name = 'Engine_id #{}, FD00{}'.format(i, dataset_number+1)
        ax.set_title(title_name)
        ax.grid(True)

        # Plotting on the first y-axis
        ax.plot(cycles, true_label, color='black', label='True_RUL')
        ax.plot(cycles, predicted, color='blue', label='predicted')
        ax.legend(loc='upper left')
        if save:

            path = './reports/figures'
            if not os.path.exists(path):
                os.makedirs(path)

            plt.savefig(os.path.join(path, title_name + '.png'))
        else:

            plt.show()

    def vis_uncertainty(self, true_label, predicted, stds, i, dataset_number, model_name, save):
        """
        Show a figure depending on the uncertainty estimation of a specific engine id
        """
        cycles = [x for x in range(len(true_label))]
        assert (len(true_label) == len(predicted))
        fig, ax = plt.subplots(figsize=(10, 6))

        # Same as above
        ax.set_xlabel('Cycles')
        ax.set_ylabel("RUL")
        if model_name == "BAYSIAN_LSTM":
            title_name = 'BAYSIAN_Engine_id #{}, FD00{}'.format(i, dataset_number + 1)
            ax.set_title(title_name)
        else:
            title_name = 'DROPOUT_Engine_id #{}, FD00{}'.format(i, dataset_number + 1)
            ax.set_title(title_name)
        ax.grid(True)

        # Plotting on the first y-axis
        ax.plot(cycles, true_label, color='black', label='True_RUL')
        ax.plot(cycles, predicted, color='blue', label='predicted')
        ax.fill_between(cycles, predicted - stds, predicted + stds, color='red', alpha=0.2, label='Uncertainty')
        ax.legend(loc='upper right')
        if save:

            path = './reports/figures'
            if not os.path.exists(path):
                os.makedirs(path)

            plt.savefig(os.path.join(path, title_name + '.png'))
        else:

            plt.show()
