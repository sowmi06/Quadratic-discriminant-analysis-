# To pull the features and labels from the dataset
import dataset as ds
import numpy as np


class QDA():
    # x is features and y is class
    def __init__(self, x_train, y_train, dataset_train, x_test, y_test, dataset_test):
        self.x_train = x_train
        self.y_train = y_train
        self.dataset_train = dataset_train
        self.x_test = x_test
        self.y_test = y_test
        self.dataset_test = dataset_test
        self.classes = self.unique_values()

def main():
    # calling dataset1 from the "dataset.py" file
    x_train, y_train, dataset_train = ds.pullDataset("./dataset1/training_validation")
    x_test, y_test, dataset_test = ds.pullDataset("./dataset1/test")

    # calling the test and train module
    qda = QDA(x_train, y_train, dataset_train, x_test, y_test, dataset_test)
    mean_k, theta_k, sigma_k = qda.k()
    qda.train_accuracy(mean_k,theta_k,sigma_k)
    qda.test_accuracy(mean_k, theta_k, sigma_k)


if __name__ == "__main__":
    main()
