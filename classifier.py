from typing import Union, Dict, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

"""
To run, ensure DATA_FOLDER points to the folder containing the uncompressed .csv data files.
DATA_SEED should remain constant to avoid cross-contamination between training and testing sets.
"""

DATA_FOLDER = "./data/"
DATA_SEED = 123
FEATURES = ["gender", "region", "highest_education", "imd_band", "age_band",
            "num_of_prev_attempts", "studied_credits", "disability"]


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


"""
Train a new decision tree model using a training set 
"""
def decision_tree(train) -> DecisionTreeClassifier:
    data = pd.get_dummies(train[FEATURES], drop_first=True)
    labels = train.final_result

    tree_cla = DecisionTreeClassifier()
    tree_cla.fit(data, labels)

    return tree_cla


def support_vector_machine(train) -> SVC:
    data = pd.get_dummies(train[FEATURES], drop_first=True)
    labels = train.final_result

    svc_cla = SVC()
    svc_cla.fit(data, labels)

    return svc_cla


"""
Output evaluation data based on a testing set and prediction function
"""
def evaluate_model(test, predictor) -> None:
    predictions = predictor(test)
    print("Accuracy score: ", accuracy_score(test.final_result, predictions))
    print(confusion_matrix(test.final_result, predictions))


if __name__ == "__main__":
    def main() -> int:
        data = load_data(DATA_FOLDER, {"studentInfo.csv", "courses.csv"})

        train, test = split_data(data["studentInfo"], 0.75, DATA_SEED)

        cla = decision_tree(train)
        evaluate_model(test, lambda t: cla.predict(pd.get_dummies(t[FEATURES], drop_first=True)))

        cla = support_vector_machine(train)
        evaluate_model(test, lambda t: cla.predict(pd.get_dummies(t[FEATURES], drop_first=True)))

        return 0


    exit(main())
