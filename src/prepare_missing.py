import pandas as pd
import numpy as np

from utils import read_yaml



def transxf(x, tcode):
    '''
    =========================================================================
    DESCRIPTION:
     This function transforms a SINGLE SERIES (in a column vector) as specified
     by a given transformation code.

     -------------------------------------------------------------------------
     INPUT:
               x       = series (in a column vector) to be transformed
               tcode   = transformation code (1-7)

     OUTPUT:
               y       = transformed series (as a column vector)
     -------------------------------------------------------------------------
    '''
    assert x.shape[1] == 1, 'x must contain one column'

    name = x.columns.values[0]
    x.rename(columns={name:'original'}, inplace=True)

    small = 1e-6                          # Value close to zero

    if tcode == 1:  # Level (i.e. no transformation): x(t)
        x[name] = x

    elif tcode == 2: # First difference: x(t)-x(t-1)
        x[name] = x.diff()

    elif tcode == 3: #  Second difference: (x(t)-x(t-1))-(x(t-1)-x(t-2))
        x[name] = x.diff().diff()

    elif tcode == 4: # Natural log: ln(x)
        if x.min()[0] > small:
            x[name] = np.log(x)

    elif tcode == 5: # First difference of natural log: ln(x)-ln(x-1)
        if x.min()[0] > small:
            x[name] = np.log(x).diff()

    elif tcode == 6: # Second difference of natural log: (ln(x)-ln(x-1))-(ln(x-1)-ln(x-2))
        if x.min()[0] > small:
            x[name] = np.log(x).diff().diff()

    elif tcode == 7: # First difference of percent change: (x(t)/x(t-1)-1)-(x(t-1)/x(t-2)-1)
        x[name] = x.pct_change().diff()

    else:
        x[name] = np.nan

    return x[name]


def prepare_missing(rawdata, tcode):
    ''' =========================================================================
     DESCRIPTION:
     This function transforms raw data based on each series' transformation
     code.

     -------------------------------------------------------------------------
     INPUT:
               rawdata     = raw data
               tcode       = transformation codes for each series

     OUTPUT:
               yt          = transformed data

     -------------------------------------------------------------------------
    SUBFUNCTION:
               transxf:    transforms a single series as specified by a
                           given transfromation code

     ========================================================================='''

    transformed_data = pd.DataFrame()
    variables = rawdata.columns.values     # get variable names

    for var in variables:
        x = rawdata[[var]].copy()
        transformed_data[var] = transxf(x, int(tcode[var]))

    return transformed_data


if __name__ == "__main__":
    data = pd.read_csv('./data/QD_2024_06_24.csv')        # read in data

    #### DATA GROUPS ####
    setting_series = read_yaml("src/settings.yaml")
    indicator_series = read_yaml("src/indicators.yaml")
    series = {**setting_series, **indicator_series}

    print(series[130822])
    tcode = {str(serie) : series[serie]['transformation'] for serie in series}

    data.set_index('fecha', inplace=True, drop=True)
    data.index.name = 'date'

    transformed_data = prepare_missing(data, tcode)
    transformed_data.to_csv('./data/QD_transformed.csv')

