from src.my_parser import MyParser



def main(config_path):
    """
    Use this script to test a trained model on each sub test-set from CMAPS
    :param config_path: config path
    :return: shows loss of each dataset
    """
    parser = MyParser(config_path)
    parser.load_data()

    test_loaders = parser.get_test_data()

    model = parser.load_model()
    i = 0
    print("started predicting..\n")
    for dataset in test_loaders:
        predictor = parser.get_predictor(dataset, model)
        mean_loss, outputs, real, stds = predictor.predict()
        parser.visualize_trajectory(i, outputs, stds, real)

        i += 1

        print("loss of dataset Test_FD00{} = {:.2f}".format(i, mean_loss))
        print("outputs shape {}".format(outputs.shape))


if __name__ == '__main__':
    path = 'config/config.txt'
    main(path)
