import pandas as pd
import numpy as np

def get_model_name_from_df(
        row: pd.Series,
) -> str:
    # receives as input a row of the dataframe
    # and, using the column aggregated_results,
    # gets the model names from the json and returns them

    model_names = row["aggregatedResults"]

    model_names = [model_name.split("_")[0] for model_name in model_names]

    return model_names

def preprocess_dataframe():
    pass