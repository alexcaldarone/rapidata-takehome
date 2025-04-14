from typing import Union, Tuple, List, Dict

from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, classification_report, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnDropper(BaseEstimator, TransformerMixin):
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
 ):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def create_column_transformer(cat_cols: List[str], num_cols: List[str]):
    # this can be extended to incluse as many column transformations as we wish
    return ColumnTransformer(
        transformers=[
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ],
        remainder="passthrough"
    )

def evaluate_model(model, X_test, y_test):
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