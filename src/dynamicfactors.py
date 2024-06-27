import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from utils import read_yaml

df = pd.read_csv("data/QD_transformed.csv")
# Drop the last row
df = df.drop(df.index[-1])

df.set_index("date", inplace=True)


#### DATA GROUPS ####
setting_series = read_yaml("src/settings.yaml")
indicator_series = read_yaml("src/indicators.yaml")
series = {**setting_series, **indicator_series}

factors = {str(serie) : ['Global', series[serie]['group']] for serie in series}
factor_multiplicities = {'Global': 1}
factor_orders = {
    'Group 1': 2,
    'Group 2': 2,
    'Group 4': 2,
    'Group 5': 2,
    'Group 6': 2,
    'Group 7': 2,
    'Group 8': 2,
    'Global': 2}

endog_m = df.loc['1990':, :]
#print(endog_m['SR17536'])

# Construct the dynamic factor model
model = sm.tsa.DynamicFactorMQ(
    endog_m,
    factors=factors, factor_orders=factor_orders,
    factor_multiplicities=factor_multiplicities)

results = model.fit(disp=10, maxiter=50)
print(results.summary())


##### R2 PLOT #####
with sns.color_palette('deep'):
    fig = results.plot_coefficients_of_determination(method='individual', figsize=(14, 9))
    fig.suptitle(r'$R^2$ - regression on individual factors', fontsize=14, fontweight=600)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



##### GLOBAL FACTOR PLOT #####

# Get estimates of the global factor,
# conditional on the full dataset ("smoothed")
factor_names = ['Global']
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



##### GDP FORECAST PLOTS#####
GDP_variable = '736181'
fcast_q = results.forecast('2026-12')[GDP_variable]
fcast_q.index.strftime('%Y-%m-%d')
print(fcast_q)

plot_q = pd.concat([df.loc['1990':, GDP_variable], fcast_q])
plot_q.index = pd.to_datetime(plot_q.index)

with sns.color_palette('deep'):
    fig, ax = plt.subplots(figsize=(14, 4))

    # Plot real GDP growth, data, and forecasts
    plot_q.plot(ax=ax)
    ax.set(title='Real Gross Domestic Product (transformed: annualized growth rate)')
    
    # Add horizontal line at zero
    ax.hlines(0, plot_q.index[0], plot_q.index[-1], linewidth=1)
    
    # Highlight the forecast period
    ylim = ax.get_ylim()
    ax.fill_between(plot_q.loc['2024-04':].index,
                    ylim[0], ylim[1], alpha=0.1, color='C0')
    
    # Annotate the forecast period
    ax.annotate(r' Forecast $\rightarrow$', 
                ('2022-01', ylim[0] + 0.1 * (ylim[1] - ylim[0])))
    ax.set_ylim(ylim)

    # Title
    fig.suptitle('Data and forecasts (January 2024 vintage), transformed scale',
                 fontsize=14, fontweight=600)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Reverse the transformations

# For real GDP, we take the level in 1990Q1 from the original data,
# and then apply the growth rates to compute the remaining levels
plot_q_orig = (plot_q / 100 + 1)**0.25
plot_q_orig.loc['1990-01-01'] = df.loc['1990-01-01', GDP_variable]
plot_q_orig = plot_q_orig.cumprod()


with sns.color_palette('deep'):
    fig, ax = plt.subplots(figsize=(14, 4))

    # Plot real GDP, data, and forecasts
    plot_q_orig.plot(ax=ax)
    ax.set(title=('Real Gross Domestic Product (in Billions)'))
    
    
    # Highlight the forecast period
    ylim = ax.get_ylim()
    ax.fill_between(plot_q_orig.loc['2024-04':].index,
                    ylim[0], ylim[1], alpha=0.1, color='C0')
    
    # Annotate the forecast period
    ax.annotate(r' Forecast $\rightarrow$', 
                ('2022-01', ylim[0] + 0.1 * (ylim[1] - ylim[0])))
    ax.set_ylim(ylim)

    # Title
    fig.suptitle('Data and forecasts (January 2024 vintage), original scale',
                 fontsize=14, fontweight=600)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()