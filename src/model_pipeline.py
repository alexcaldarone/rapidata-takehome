from typing import Union, Tuple, List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import auc, precision_recall_curve, classification_report, roc_curve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Defines a custom transformer to drop specified columns from a DataFrame.ì,
    by inheriting from BaseEstimator and TransformerMixin (sklearn base classes).

    Parameters
    ----------
    columns_to_drop : list
        List of column names to drop from the DataFrame.
    """
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, axis=1)

def train_test_split_stratified(
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = 0.25, 
        random_state: int =42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs a stratified train-test split on the given DataFrame and Series.

    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Target Series.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
            - X_train: Training features DataFrame.
            - X_test: Testing features DataFrame.
            - y_train: Training target Series.
            - y_test: Testing target Series.
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def create_column_transformer(
        cat_cols: List[str], 
        num_cols: List[str]
) -> ColumnTransformer:
    """
    Creates a column transformer for preprocessing categorical and numerical features.

    Args:
        cat_cols (List[str]): List of categorical column names.
        num_cols (List[str]): List of numerical column names.
    
    Returns:
        ColumnTransformer: A column transformer that applies transformations to data.
    """
    return ColumnTransformer(
        transformers=[
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ],
        remainder="passthrough"
    )

def evaluate_model(
    model: Pipeline, 
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Computes evaluation metrics for the given model on the test set.

    Args:
        model (Pipeline): The trained model pipeline.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): True labels for the test set.
    
    Returns:
        Dict[str, Union[float, Dict[str, float]]]: 
            - fpr: False Positive Rate.
            - tpr: True Positive Rate.
            - roc_auc: Area Under the ROC Curve.
            - precision: Precision scores.
            - recall: Recall scores.
            - pr_auc: Area Under the Precision-Recall Curve.
            - report: Classification report as a dictionary.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # for binary classification

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
        "report": report
    }

def plot_roc_curves(results_dict):
    """
    Plots ROC curves for multiple models.
    """
    plt.figure(figsize=(8, 6))
    for name, res in results_dict.items():
        plt.plot(res["fpr"], res["tpr"], label=f"{name} (AUC = {res['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_pr_curves(results_dict):
    """
    Plots Precision-Recall curves for multiple models.
    """
    plt.figure(figsize=(8, 6))
    for name, res in results_dict.items():
        plt.plot(res["recall"], res["precision"], label=f"{name} (AUC = {res['pr_auc']:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()