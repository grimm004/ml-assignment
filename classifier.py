from typing import Union, Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import OneHotEncoder

pd.options.mode.chained_assignment = None

"""
To run, ensure DATA_FOLDER points to the folder containing the uncompressed .csv data files.
DATA_SEED should remain constant to avoid cross-contamination between training and testing sets.
Change OUTPUTS to toggle the various outputs
Change METHODS to toggle the methods being run
"""

DATA_FOLDER: str = "./datasets/"
DATA_SEED: int = 123
FEATURES: List[str] = ["gender", "region", "highest_education", "imd_band", "age_band",
                       "num_of_prev_attempts", "studied_credits", "disability"]
OUTPUTS = {
    "set_ratios": True,
    "accuracy_scores": True,
    "confusion_matrices": True,
}

METHODS = {
    "decision_tree": False,
    "support_vector": False
}


"""
Load a set of CSV data tables into a DataFrame dictionary
"""
def load_data(folder: str, source_set: Iterable[str]) -> Dict[str, pd.DataFrame]:
    return {source.replace(".csv", ""): pd.read_csv(os.path.join(folder, source)) for source in source_set}


"""
Randomly split the provided data into a training set (proportional to train_ratio) and a testing set.
"""
def split_data(data, train_ratio: float, seed: Union[int, None] = None):
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(len(data))
    train_size = int(len(data) * train_ratio)
    return data.iloc[indices[:train_size]], data.iloc[indices[train_size:]]


def decision_tree(categories, dt_train, dt_test):
    # Fetch the training features and labels from the training set
    train_features = pd.get_dummies(dt_train[FEATURES], drop_first=True)
    train_labels = dt_train.final_result

    # Fetch the testing features and labels from the testing set
    test_features = pd.get_dummies(dt_test[FEATURES], drop_first=True)
    test_labels = dt_test.final_result

    # Create and train a decision tree classifier on the training set
    dt_cla = DecisionTreeClassifier()
    dt_cla.fit(train_features, train_labels)

    if OUTPUTS["accuracy_scores"]:
        print("DT Score: ", dt_cla.score(test_features, test_labels))

    if OUTPUTS["confusion_matrices"]:
        # Calculate and output a confusion matrix for the decision tree based on the test data
        plot_confusion_matrix(dt_cla, test_features, test_labels,
                              display_labels=categories,
                              cmap=plt.cm.Blues,
                              normalize="true")
        plt.show()


def support_vector(categories, sv_train, sv_test):
    # Fetch the training features and labels from the training set
    train_features = pd.get_dummies(sv_train[FEATURES], drop_first=True)
    train_labels = sv_train.final_result

    # Fetch the testing features and labels from the testing set
    test_features = pd.get_dummies(sv_test[FEATURES], drop_first=True)
    test_labels = sv_test.final_result

    # Create and train a support vector classifier on the training set
    svc_cla = SVC()
    svc_cla.fit(train_features, train_labels)

    if OUTPUTS["accuracy_scores"]:
        print("SVM Score: ", svc_cla.score(test_features, test_labels))

    if OUTPUTS["confusion_matrices"]:
        # Calculate and output a confusion matrix for the decision tree based on the test data
        plot_confusion_matrix(svc_cla, test_features, test_labels,
                              display_labels=categories,
                              cmap=plt.cm.Reds,
                              normalize="true")
        plt.show()


if __name__ == "__main__":
    def main() -> int:
        data_sets = load_data(DATA_FOLDER, {"studentInfo.csv", "courses.csv"})

        # Transform table data to sample data
        raw_data = data_sets["studentInfo"]

        # Remove rows with the 'Withdrawn' final_result value
        data = raw_data[raw_data.final_result != "Withdrawn"]

        # Make IMD band data consistent
        data.imd_band.replace("10-20", "10-20%", inplace=True)

        # Split sample data in to training and testing set
        train, test = split_data(data, 0.75, DATA_SEED)

        if OUTPUTS["set_ratios"]:
            # Output a stacked bar chart showing the outcome distributions within the two datasets
            chart = pd.DataFrame(columns=["Fail", "Pass", "Distinction"],
                                 index=["All", "Train", "Test"],
                                 data=[
                                     100 * data["final_result"].value_counts(normalize=True),
                                     100 * train["final_result"].value_counts(normalize=True),
                                     100 * test["final_result"].value_counts(normalize=True)
                                 ]).plot.bar(stacked=True)
            chart.set_xlabel("Dataset")
            chart.set_ylabel("Outcome Distribution (Percentage)")
            plt.show()

        categories = data.final_result.unique()

        if METHODS["decision_tree"]:
            decision_tree(categories, train.copy(), test.copy())
        if METHODS["support_vector"]:
            support_vector(categories, train.copy(), test.copy())

        return 0


    exit(main())
