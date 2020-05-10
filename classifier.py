from typing import Union, Dict, Iterable, Tuple, Set
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

pd.options.mode.chained_assignment = None
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

"""
NumPy, pandas, sklearn and matplotlib are required.
To run, ensure DATA_FOLDER points to the folder containing the uncompressed .csv data files.
Either launch from IDLE or the console, the first output is the dataset ratios (if enabled),
next the random forest accuracy score and confusion matrix are output and finally the support-
vector machine accuracy score and confusion matrix are output.
DATA_SEED should remain constant to avoid cross-contamination between training and testing sets.
Change OUTPUTS to toggle the various outputs
Change METHODS to toggle the methods being run
"""

DATA_FOLDER: str = "./datasets/"
DATA_SEED: int = 123

FEATURES: Dict[str, Union[Set[str], Dict[str, Dict[str, int]]]] = {
    # Continuous data
    "con": {
        "num_of_prev_attempts",
        "studied_credits",
        "predicted_score"
    },
    # Unordered/un-mappable categorical data
    "cat": {
        "region"
    },
    # Ordered/mappable categorical data
    "cat_map": {
        "highest_education": {
            "No Formal quals": 0,
            "Lower Than A Level": 1,
            "A Level or Equivalent": 2,
            "HE Qualification": 3,
            "Post Graduate Qualification": 4
        },
        "gender": {"M": 0, "F": 1},
        "disability": {"Y": 0, "N": 1},
        # Take mid-points of imd_band ranges
        "imd_band": {
            "0-10%": 5,
            "10-20": 15,
            "20-30%": 25,
            "30-40%": 35,
            "40-50%": 45,
            "50-60%": 55,
            "60-70%": 65,
            "70-80%": 75,
            "80-90%": 85,
            "90-100%": 95
        },
        # Take min of age_band ranges
        "age_band": {
            "0-35": 0,
            "35-55": 35,
            "55<=": 55
        }
    }
}

# ML method control and hyperparameters
METHODS: Dict[str, Dict[str, Union[bool, float, int, str]]] = {
    "random_forest": {
        "enabled": True,
        "random_seed": 321,
        "estimator_count": 100,
        "max_depth": None,
        "min_leaf_samples": 1
    },
    "support_vector": {
        "enabled": True,
        "c_value": 1.0,
        "class_comparator": "ovr",
        "kernel": "rbf"
    }
}

OUTPUTS: Dict[str, bool] = {
    "set_ratios": True,
    "accuracy_scores": True,
    "confusion_matrices": True
}


def load_data(folder: str, source_set: Iterable[str]) -> Dict[str, pd.DataFrame]:
    """
    Load a set of CSV data tables into a DataFrame dictionary

    :param folder: The folder to load tables from
    :param source_set: A set containing the file names to load
    :return: A dictionary mapping the table name to the loaded DataFrame
    """

    return {source.replace(".csv", ""): pd.read_csv(os.path.join(folder, source)) for source in source_set}


def split_data(data, train_ratio: float, seed: Union[int, None] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Randomly split the provided data into a training set (proportional to train_ratio) and a testing set.

    :param data: The dataset to split
    :param train_ratio: The fraction of training data
    :param seed: A seed for the numpy random module
    :return: A tuple containing the testing and training datasets
    """

    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(len(data))
    train_size = int(len(data) * train_ratio)
    return data.iloc[indices[:train_size]], data.iloc[indices[train_size:]]


def train_and_test_model(model, train, test, cmap) -> None:
    """
    Train and test a model

    :param model: The model to train and test
    :param train: A tuple containing the training samples and labels
    :param test: A tuple contain the testing samples and labels
    :param cmap: The colour map for the confusion matrix
    :return: None
    """

    train_samples, train_labels = train
    test_samples, test_labels = test

    # Train the classifier
    model.fit(train_samples, train_labels)

    if OUTPUTS["accuracy_scores"]:
        # Output the model's score when run on the test data
        print("Score: ", model.score(test_samples, test_labels))

    if OUTPUTS["confusion_matrices"]:
        # Calculate and output a confusion matrix based on the test data
        plot_confusion_matrix(model, test_samples, test_labels,
                              cmap=cmap,
                              normalize="true")
        plt.show()


if __name__ == "__main__":
    def main() -> int:
        # Load the required tables
        data_sets = load_data(DATA_FOLDER, {"studentInfo.csv", "studentAssessment.csv", "assessments.csv"})

        # Transform table data to sample data
        student_df = data_sets["studentInfo"].set_index(["code_module", "code_presentation", "id_student"], drop=False)
        student_assessment_df = data_sets["studentAssessment"]
        assessment_df = data_sets["assessments"]

        # Calculate weighted assessment scores per module-presentation-student
        merged = student_assessment_df.merge(assessment_df, on=["id_assessment"])
        merged["weighted_score"] = merged["score"] * merged["weight"] / 100
        merged.drop(["score", "weight", "date", "date_submitted", "is_banked"], axis="columns", inplace=True)
        predicted_scores = merged.groupby(["code_module", "code_presentation", "id_student"])["weighted_score"].sum()

        # Remove rows with the 'Withdrawn' final_result value
        data = student_df[student_df["final_result"] != "Withdrawn"]

        # Add median score
        data["predicted_score"] = predicted_scores

        # Drop rows with undefined column values
        data.dropna(how="any", inplace=True)

        # Map categorical features to numerical values
        for feature, feature_map in FEATURES["cat_map"].items():
            data[feature] = data[feature].map(feature_map)

        features = FEATURES["con"] | FEATURES["cat_map"].keys()
        if len(FEATURES["cat"]) > 0:
            # One-hot encode un-mappable categorical features
            one_hot = pd.get_dummies(data[FEATURES["cat"]], drop_first=True)
            data = data.join(one_hot)
            data.drop(FEATURES["cat"], "columns")
            features |= set(one_hot.columns)

        assert len(features) > 0

        print("Using feature set: ", features)

        # Split sample data in to training and testing set
        train, test = split_data(data[list(features) + ["final_result"]], 0.75, DATA_SEED)

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

        # Fetch the training features and labels from the training set
        train_set = train[features], train["final_result"]

        # Fetch the testing features and labels from the testing set
        test_set = test[features], test["final_result"]

        rf_options = METHODS["random_forest"]
        if rf_options["enabled"]:
            # Create, train and test the Random Forest classifier
            rf = RandomForestClassifier(
                n_estimators=rf_options["estimator_count"],
                random_state=rf_options["random_seed"],
                max_depth=rf_options["max_depth"],
                min_samples_leaf=rf_options["min_leaf_samples"]
            )
            train_and_test_model(rf, train_set, test_set, "Blues")
        sv_options = METHODS["support_vector"]
        if sv_options["enabled"]:
            # Create, train and test the Support-Vector classifier
            sv = SVC(
                sv_options["c_value"], sv_options["kernel"],
                decision_function_shape=sv_options["class_comparator"]
            )
            train_and_test_model(sv, train_set, test_set, "Reds")

        return 0


    exit(main())
