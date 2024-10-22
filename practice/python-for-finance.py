import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from mplfinance.original_flavor import candlestick_ohlc
from matplotlib.dates import date2num
import pandas as pd
from yahoo_fin import stock_info as si

style.use('ggplot')

# Load data from CSV
df = pd.read_csv("HDFCBANK.csv", parse_dates=True, index_col=0)

# Convert index to a regular column
df.reset_index(inplace=True)
df.rename(columns={'index': 'Date'}, inplace=True)

# Resample data to 10-day intervals for OHLC and volume
df_ohlc = df.set_index('Date')['adjclose'].resample('10D').ohlc()
df_volume = df.set_index('Date')['volume'].resample('10D').sum()

# Reset the index to make 'Date' a column, convert to mdates format for candlestick_ohlc
df_ohlc.reset_index(inplace=True)
df_ohlc['Date'] = df_ohlc['Date'].map(date2num)
print(df_ohlc.head())

# Set up subplots
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)

# Create candlestick chart on ax1
candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')

# Plot volume on ax2
ax2.fill_between(df_volume.index.map(date2num), df_volume.values, 0)

plt.show()
