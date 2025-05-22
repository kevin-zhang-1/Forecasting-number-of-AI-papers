import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Load data and parse dates
df = pd.read_csv('arxiv_cs_ai_monthly_counts.csv', parse_dates=['Month'])
df.columns = ['Month', 'Papers']

current_date = pd.Timestamp.now().normalize()
df = df[df['Month'] <= current_date]

# Handle zeros in 'Papers' before log transform
df['Log_Papers'] = np.log(df['Papers'] + 1)  # Add 1 to avoid log(0)

# Extract month number
df['Month_Number'] = df['Month'].dt.month

# Calculate and merge seasonal averages
monthly_avg = df.groupby('Month_Number')['Log_Papers'].mean().reset_index()
monthly_avg.columns = ['Month_Number', 'Seasonal_Avg']
df = df.merge(monthly_avg, on='Month_Number', how='left')

# Deseasonalize and clean data
df['Deseasonalized'] = df['Log_Papers'] - df['Seasonal_Avg']
df = df.replace([np.inf, -np.inf], np.nan)  # Replace infinities
df = df.dropna(subset=['Deseasonalized'])   # Drop NaN rows

# ADF Test
result = adfuller(df['Deseasonalized'])
print(f'ADF Statistic: {result[0]:.3f}')
print(f'p-value: {result[1]:.3f}')

df['Deseasonalized_diff'] = df['Deseasonalized'].diff(1).dropna()

result_diff = adfuller(df['Deseasonalized_diff'].dropna())
print(f'ADF Statistic (d=1): {result_diff[0]:.3f}')
print(f'p-value (d=1): {result_diff[1]:.3f}')

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(df['Deseasonalized_diff'].dropna(), lags=24, ax=ax1)
plot_pacf(df['Deseasonalized_diff'].dropna(), lags=24, ax=ax2, method='ywm')
plt.tight_layout()
plt.show()


import itertools
import statsmodels.api as sm

p = range(0, 3)  # AR order
d = range(0, 2)  # Differencing
q = range(0, 3)  # MA order

best_aicc = np.inf
best_order = None

for order in itertools.product(p, d, q):
    try:
        model = sm.tsa.ARIMA(df['Deseasonalized'], order=order)
        results = model.fit()
        if results.aicc < best_aicc:
            best_aicc = results.aicc
            best_order = order
    except:
        continue

print(f'Best ARIMA Order: {best_order}, AICC: {best_aicc}')

model = sm.tsa.ARIMA(df['Deseasonalized'], order=(best_order))
results = model.fit()
print(results.summary())

# Ljung-Box Test
from statsmodels.stats.diagnostic import acorr_ljungbox

lb_test = acorr_ljungbox(results.resid, lags=20)
print(f'Ljung-Box p-values: {lb_test.lb_pvalue}')  # All > 0.05 → No autocorrelation

# Residual ACF/PACF
plot_acf(results.resid, lags=40)
plot_pacf(results.resid, lags=40)
plt.show()

forecast_steps = 50
forecast = results.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

import pandas as pd

# Generate future dates (monthly frequency starting after last observed date)
last_date = df['Month'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')

# Create DataFrame for future months
future_df = pd.DataFrame({'Month': future_dates})
future_df['Month_Number'] = future_df['Month'].dt.month

# Add seasonal averages to future dates
future_df = future_df.merge(monthly_avg, on='Month_Number', how='left')


# Add seasonal component back to deseasonalized forecasts
forecast_mean_reverted = forecast_mean + future_df['Seasonal_Avg'].values
conf_int_reverted_lower = conf_int.iloc[:, 0] + future_df['Seasonal_Avg'].values
conf_int_reverted_upper = conf_int.iloc[:, 1] + future_df['Seasonal_Avg'].values

# Exponentiate and subtract 1 (reverse log(x+1))
forecast_papers = np.exp(forecast_mean_reverted) - 1
conf_int_lower_papers = np.exp(conf_int_reverted_lower) - 1
conf_int_upper_papers = np.exp(conf_int_reverted_upper) - 1

plt.figure(figsize=(14, 6))

# Plot observed data
plt.plot(df['Month'], df['Papers'], label='Observed', color='steelblue')

# Plot forecasts
plt.plot(future_dates, forecast_papers, color='crimson', label='ARIMA Forecast')

# Plot confidence intervals
plt.fill_between(future_dates, 
                 conf_int_lower_papers, 
                 conf_int_upper_papers, 
                 color='lightcoral', alpha=0.3, label='95% CI')

# Formatting
plt.title('ARIMA(0,1,1) Forecast for arXiv cs.AI Papers\nWith 95% Confidence Intervals', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Monthly Papers')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# Create DataFrame with forecasts
forecast_df = pd.DataFrame({
    'Month': future_dates,
    'Forecast_Papers': forecast_papers,
    'CI_Lower': conf_int_lower_papers,
    'CI_Upper': conf_int_upper_papers
})

# Merge with historical data (optional)
full_df = pd.concat([
    df[['Month', 'Papers']].assign(Type='Observed'),
    forecast_df.assign(Type='Forecast')
], ignore_index=True)

# Save to CSV
forecast_df.to_csv('arxiv_cs_ai_forecasts.csv', index=False)
print("Forecasts saved to arxiv_cs_ai_forecasts.csv")