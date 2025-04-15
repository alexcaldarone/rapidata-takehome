import pandas as pd
import numpy as np
import pycountry_convert as pc

def get_model_name_from_df(
        row: pd.Series,
) -> str:
    """
    Gets the names of the models compared in the study from the DataFrame.

    Args:
        row (pd.Series): A row from the DataFrame containing the model names.
    
    Returns:
        str: The name of the model.
    """
    model_names = row["aggregatedResults"]

    model_names = [model_name.split("_")[0] for model_name in model_names]

    return model_names

def country_code_to_continent(code: str) -> str:
    """
    Converts a country code to its continent name.

    Args:
        code (str): The country code.
    
    Returns:
        str: The continent name.
    """
    try:
        code = code.upper()
        continent = pc.country_alpha2_to_continent_code(code)
        continent = pc.convert_continent_code_to_continent_name(continent)
        return continent
    except:
        return "Unknown"