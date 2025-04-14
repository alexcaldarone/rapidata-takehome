import pandas as pd
import numpy as np
import pycountry_convert as pc

def get_model_name_from_df(
        row: pd.Series,
) -> str:
    # receives as input a row of the dataframe
    # and, using the column aggregated_results,
    # gets the model names from the json and returns them

    model_names = row["aggregatedResults"]

    model_names = [model_name.split("_")[0] for model_name in model_names]

    return model_names

def country_code_to_continent(code: str) -> str:
    try:
        code = code.upper()
        continent = pc.country_alpha2_to_continent_code(code)
        continent = pc.convert_continent_code_to_continent_name(continent)
        return continent
    except:
        return "Unknown"
