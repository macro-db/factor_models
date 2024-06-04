import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from utils import read_yaml

df = pd.read_csv("data/2019-07-transformed.csv")
df.set_index("date", inplace=True)


#### DATA GROUPS ####
setting_series = read_yaml("src/settings.yaml")
indicator_series = read_yaml("src/indicators.yaml")
series = {**setting_series, **indicator_series}

factors = {str(serie) : ['Global', series[serie]['group']] for serie in series}
factor_multiplicities = {'Global': 2}
factor_orders = {
    'Group 1': 1,
    'Group 2': 1,
    'Group 4': 1,
    'Group 5': 1,
    'Group 6': 1,
    'Group 7': 1,
    'Group 8': 1,
    'Global': 4}

endog_m = df.loc['1990':, :]
#print(endog_m['SR17536'])

# Construct the dynamic factor model
model = sm.tsa.DynamicFactorMQ(
    endog_m,
    factors=factors, factor_orders=factor_orders,
    factor_multiplicities=factor_multiplicities)

#print(model.summary())

results = model.fit(disp=10, maxiter=50)
print(results.summary())



with sns.color_palette('deep'):
    fig = results.plot_coefficients_of_determination(method='individual', figsize=(14, 9))
    fig.suptitle(r'$R^2$ - regression on individual factors', fontsize=14, fontweight=600)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Get estimates of the global and labor market factors,
# conditional on the full dataset ("smoothed")
factor_names = ['Global.1', 'Global.2']
mean = results.factors.smoothed[factor_names]

# Compute 95% confidence intervals
from scipy.stats import norm
std = pd.concat([results.factors.smoothed_cov.loc[name, name]
                 for name in factor_names], axis=1)
crit = norm.ppf(1 - 0.05 / 2)
lower = mean - crit * std
upper = mean + crit * std

with sns.color_palette('deep'):
    fig, ax = plt.subplots(figsize=(14, 3))
    mean.plot(ax=ax)
    
    for name in factor_names:
        ax.fill_between(mean.index, lower[name], upper[name], alpha=0.3)
    
    ax.set(title='Estimated factors: smoothed estimates and 95% confidence intervals')
    fig.tight_layout()
    plt.show()


    # Create point forecasts, 3 steps ahead
point_forecasts = results.forecast(steps=3)

# Print the forecasts for the first 5 observed variables
print(point_forecasts)