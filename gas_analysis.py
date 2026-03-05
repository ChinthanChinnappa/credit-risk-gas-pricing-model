import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# --------------------------------------------------
# 1. LOAD AND CLEAN DATA
# --------------------------------------------------
df = pd.read_csv("natural_gas_prices.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values("Date")
df.set_index("Date", inplace=True)

print(df.head())
print(df.info())

# --------------------------------------------------
# 2. VISUALIZE HISTORICAL DATA
# --------------------------------------------------
plt.figure()
plt.plot(df.index, df['Price'])
plt.title("Natural Gas Price Trend")
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 3. CHECK STATIONARITY
# --------------------------------------------------
adf_result = adfuller(df['Price'])
print("ADF p-value:", adf_result[1])

# --------------------------------------------------
# 4. SEASONAL DECOMPOSITION
# --------------------------------------------------
decomposition = seasonal_decompose(df['Price'], model='additive', period=12)
decomposition.plot()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 5. BUILD SARIMA MODEL
# --------------------------------------------------
model = SARIMAX(df['Price'],
                order=(1,1,1),
                seasonal_order=(1,1,1,12),
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit()

# --------------------------------------------------
# 6. FORECAST NEXT 12 MONTHS
# --------------------------------------------------
forecast_steps = 12
forecast = results.forecast(steps=forecast_steps)

future_dates = pd.date_range(start=df.index.max() + pd.offsets.MonthEnd(1),
                             periods=forecast_steps,
                             freq='M')

forecast_df = pd.DataFrame({
    'Price': forecast
}, index=future_dates)

# --------------------------------------------------
# 7. PLOT HISTORICAL + FORECAST
# --------------------------------------------------
plt.figure()
plt.plot(df.index, df['Price'], label="Historical")
plt.plot(forecast_df.index, forecast_df['Price'], label="Forecast")
plt.title("Natural Gas Price Forecast (1 Year)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 8. INTERPOLATION + FORECAST FUNCTION
# --------------------------------------------------
def get_price(input_date):
    input_date = pd.to_datetime(input_date)

    # Historical or within dataset
    if input_date <= df.index.max():
        temp = df.copy()
        temp = temp.reindex(temp.index.union([input_date])).sort_index()
        temp['Price'] = temp['Price'].interpolate(method='time')
        return float(temp.loc[input_date]['Price'])

    # Future date
    else:
        months_ahead = ((input_date.year - df.index.max().year) * 12 +
                        (input_date.month - df.index.max().month))

        if months_ahead <= 0:
            months_ahead = 1

        future_forecast = results.forecast(steps=months_ahead)
        return float(future_forecast.iloc[-1])


# --------------------------------------------------
# 9. TEST FUNCTION
# --------------------------------------------------
print("Price on 2022-05-15:", get_price("2022-05-15"))
print("Price on 2025-06-30:", get_price("2025-06-30"))