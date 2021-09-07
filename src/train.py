from src import *
from src.my_parser import MyParser

#TODO save parserconfig and results after finishing
def main(config_path):
    """
    Train a model using the train dataset and split it into train and validation
    :param config_path: config path where you put the parameters of the training
    :return:
    """
    parser = MyParser(config_path)
    parser.load_data()
    train, val = parser.get_train_data()

    model = parser.get_model()

    trainer = parser.get_trainer(train, val, model)
    train_loss, val_loss = trainer.fit()
    parser.save_model(model)
    parser.save_loss(train_loss, val_loss)
    parser.save_details()


if __name__ == '__main__':
    path = 'config/config.txt'
    main(path)
