import torch as t


class Trainer:
    """
    This class will handle the training and validation steps of required model
    """

    def __init__(self, train, val, model, loss, optimizer, epochs, gpu=False, early_stopping=100,
                 step_size=200, gamma=0.1, batch_size=512):
        """

        :param train: training dataset
        :param val: validatoin dataset
        :param model: initialized model
        :param loss: loss fucntion from pytroch
        :param optimizer: chosen optimizer from pytroch
        :param epochs: number of epochs to train
        :param iterations: iterations per epoch (needed to iterate over all dataset)
               usually = num_data/batch_size*sequence_size
        :param gpu: to use gpu or not
        :param early_stopping: number of iterations to stop after val loss start increasing or stop getting better
        """
        self.train = train
        self.val = val
        self.model = model
        self.loss = loss
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.sceduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        self.epochs = epochs
        self.gpu = gpu
        self.early_stopping = early_stopping

        self.device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        self.step_size = step_size

        t.autograd.set_detect_anomaly(True)
        if gpu:
            self.model = model.cuda()
            self.loss = loss.cuda()

    def fit(self):
        """
        Train the model for the number of epochs specified
        :return: validation and training loss
        """
        print("Strated training..")
        losses_train = []
        losses_val = []
        n = 0
        epoch = 0
        while True:
            if t.cuda.is_available():
                t.cuda.empty_cache()
            if epoch == self.epochs:
                break
            print("\tTraining epoch {} from {}".format(epoch + 1, self.epochs))

            if epoch % self.step_size == 0:
                print("Current learning rate is {}".format(self.optimizer.param_groups[0]['lr']))

            loss_train = self.train_epoch()
            losses_train.append(loss_train)
            self.sceduler.step()
            loss_val = self.val_test()
            losses_val.append(loss_val)
            if epoch == 0:
                pass
            else:

                if epoch % self.early_stopping == 0:
                    if losses_val[n] <= loss_val:
                        print("Early stopping activated")
                        break
                    else:
                        n = epoch
            epoch += 1
        return losses_train, losses_val

    def val_test(self):
        """
        Iterate through the validaton dataset and validate the model
        :param iterations: number of iterations needed to go through the entire set.
        :return: validation loss
        """
        self.model.eval()
        predictions = []
        true_labels = []
        losses = []
        step = 0
        print("Total Val steps are {} ".format(len(self.val)))
        with t.no_grad():
            for data in self.val:

                if t.cuda.is_available():

                    sequence, label = data[0].cuda(), data[1].cuda()
                else:
                    sequence, label = data[0], data[1]
                if sequence.shape[0] < self.batch_size:
                    continue
                loss, output = self.val_test_step(sequence, label)
                losses.append(loss)
                predictions.append(output)
                true_labels.append(label)
                step += 1
                if step % 50 == 0:
                    print("Step: {}".format(step))

        val_loss = t.mean(t.stack(losses))
        print("Validation Loss: {:.3f}".format(val_loss))
        # f1, acc = self.evaluation_metric(predictions, true_labels)
        # # TODO add evaluation metrics after each epoch
        # print("\nF1 score is: {:.4f}, validation accuracy: {:.4f}%".format(f1, acc))
        # # print(f"\nEvaluation accuracy: {self.evaluation_metric(predictions)}")

        return val_loss

    def val_test_step(self, x, y):
        """
        actual propagation of the inputs through the model and calcualing the loss of one batch
        :param x: one batch of inputs
        :param y: one batch of true values
        :return: loss and predicted values
        """
        outputs = self.model(x)
        loss = t.sqrt(self.loss(outputs, y))
        return loss, outputs

    def train_step(self, x, y):
        """
        train one batch of train data and updated the parameters according to the loss
        :param x: sensor measurmentes of shape [batch, sequnece_length, feature_length]
        :param y:
        :return:
        """
        self.optimizer.zero_grad()
        if self.model.get_name() == "BAYSIAN_LSTM":

            mse = self.model.sample_elbo(inputs=x, labels=y, criterion=self.loss, sample_nbr=1,
                                         complexity_cost_weight=1 / (len(self.train)*self.batch_size))
            loss = t.sqrt(mse)
        else:

            outputs = self.model(x)

            loss = t.sqrt(self.loss(outputs, y))
        loss.backward()
        self.optimizer.step()
        return loss

    def train_epoch(self):
        """
        Train for one epoch
        :param iterations: number of iterations to pass through teh entire dataset once
        :return: train loss
        """
        losses = []
        step = 0
        self.model.train()
        print("Total train steps {}".format(len(self.train)))

        for data in self.train:

            if t.cuda.is_available():

                sequence, label = data[0].cuda(), data[1].cuda()
            else:
                sequence, label = data[0], data[1]

            if sequence.shape[0] < self.batch_size:
                continue

            loss = self.train_step(sequence, label)
            losses.append(loss)
            step += 1
            if step % 100 == 0:
                print("Step: {}".format(step))

        train_loss = t.mean(t.stack(losses))
        print("Training loss: {:.3f}".format(train_loss))
        return train_loss
