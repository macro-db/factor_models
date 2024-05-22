import datetime
import os
import random

import numpy as np
import pandas as pd
import requests
import yaml

BANXICO_API_KEY = os.environ.get("BANXICO_API_KEY")

def data_from_banxico(series):
    # Create session and add necessary headers
    session = requests.Session()
    headers = {"Bmx-Token": BANXICO_API_KEY, "Accept": "application/json"}
    # Add desired series to request
    url = f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{series}/datos"
    response = session.get(url, headers=headers).json()
    # Return the response
    df = pd.DataFrame(response["bmx"]["series"][0]["datos"])
    df.rename(columns={"dato": series}, inplace=True)

    return df


INEGI_API_KEY = os.environ.get("INEGI_API_KEY")


def data_from_inegi(series):
    url = f"https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/INDICATOR/{series}/es/0700/false/BIE/2.0/{INEGI_API_KEY}?type=json"
    session = requests.Session()
    response = session.get(url).json()

    df = pd.DataFrame(response["Series"][0]["OBSERVATIONS"])[
        ["TIME_PERIOD", "OBS_VALUE"]
    ]

    # Convert TIME_PERIOD column to datetime
    df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"], format="%Y/%m")

    # Format datetime as 'DD/MM/YYYY'
    df["TIME_PERIOD"] = df["TIME_PERIOD"].dt.strftime("%d/%m/%Y")

    df = df.rename(columns={"TIME_PERIOD": "fecha", "OBS_VALUE": series})

    return df


def read_yaml(yaml_file):
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)
    return data


def remove_commas(number):
    return str(number).replace(",", "")


def remove_ne(string):
    if pd.isna(string):
        return string
    else:
        return string.replace("N/E", "")


def is_outdated(date):
    date_month = date.split("/")[1]

    actual_date = datetime.date.today().replace(day=1)
    three_months_ago = (actual_date - datetime.timedelta(days=95)).strftime("%m")
    recent_months = [(int(three_months_ago) + i) % 12 or 12 for i in range(5)]

    return date_month not in recent_months


# Function to remove leading and trailing NaNs from a series
def remove_leading_trailing_nans(series):
    start_index = series.first_valid_index()
    end_index = series.last_valid_index()
    return series.loc[start_index:end_index]


def transform(column, transformation):
    # For quarterly data like GDP, we will compute
    # annualized percent changes
    # mult = 4 if column.index.freqstr[0] == 'Q' else 1
    mult = 1

    # 1 => No transformation
    if transformation == 1:
        pass
    # 2 => First difference
    elif transformation == 2:
        column = column.diff()
    # 3 => Second difference
    elif transformation == 3:
        column = column.diff().diff()
    # 4 => Log
    elif transformation == 4:
        column = np.log(column)
    # 5 => Log first difference, multiplied by 100
    #      (i.e. approximate percent change)
    #      with optional multiplier for annualization
    elif transformation == 5:
        column = np.log(column).diff() * 100 * mult
    # 6 => Log second difference, multiplied by 100
    #      with optional multiplier for annualization
    elif transformation == 6:
        column = np.log(column).diff().diff() * 100 * mult
    # 7 => Exact percent change, multiplied by 100
    #      with optional annualization
    elif transformation == 7:
        column = ((column / column.shift(1)) ** mult - 1.0) * 100

    return column
