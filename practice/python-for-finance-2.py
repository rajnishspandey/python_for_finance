from nsepython import nsefetch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from mplfinance.original_flavor import candlestick_ohlc
import os
from yahoo_fin import stock_info as si
from datetime import datetime, timedelta
import numpy as np

style.use('ggplot')

# # Fetch Nifty 500 data
def save_nifty500_tickers():
    positions = nsefetch('https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500')
    df = pd.DataFrame(positions['data'])
    symbols = df[df['symbol'] != 'NIFTY 500']['symbol']
    symbols = symbols + '.NS'
    # Save to CSV
    # df.to_csv('nifty500_data.csv', index=False)
    symbols.to_csv('nifty500_symbols.csv', index=False, header=['symbol'])
    print("Data saved to nifty500_data.csv")

# save_nifty500_tickers()

def get_data_from_yahoo():
    # Read the symbols from the CSV file
    symbols_df = pd.read_csv('nifty500_symbols.csv')
    
    # Create the stocks_dfs directory if it doesn't exist
    os.makedirs('stocks_dfs', exist_ok=True)
    
    # Set the date range for fetching data (e.g., last 5 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    # Fetch and save data for each symbol
    for symbol in symbols_df['symbol']:
        try:
            yahoo_symbol = f"{symbol}"
            
            # Fetch data using yahoo_fin
            df = si.get_data(yahoo_symbol, start_date=start_date, end_date=end_date)
            
            # Save the data to a CSV file
            file_path = os.path.join('stocks_dfs', f'{symbol}.csv')
            df.to_csv(file_path)
            print(f"Data for {symbol} saved to {file_path}")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")

# Call the function to fetch and save stock data
# get_data_from_yahoo()


def compile_data():
    # Read the symbols from the CSV file
    symbols_df = pd.read_csv('nifty500_symbols.csv')
    symbols = symbols_df['symbol'].tolist()

    # Initialize an empty DataFrame to store the combined data
    combined_data = pd.DataFrame()

    # Counter for processed symbols
    processed_count = 0

    for symbol in symbols:
        file_path = os.path.join('stocks_dfs', f'{symbol}.csv')
        if os.path.exists(file_path):
            # Read the CSV file
            df = pd.read_csv(file_path, parse_dates=True).reset_index().rename(columns={'index': 'Date'})
            
            # Extract only the 'close' column and rename it to the symbol
            close_prices = df['adjclose'].rename(symbol)
            
            # Join with the combined data
            if combined_data.empty:
                combined_data = close_prices.to_frame()
            else:
                combined_data = combined_data.join(close_prices, how='outer')
            
            processed_count += 1
            
            # Print count if it's a multiple of 10
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} symbols")

    # Save the combined data to a new CSV file
    combined_data.to_csv('nifty500_joined_closes.csv')
    print(f"Combined data saved to nifty500_joined_closes.csv")
    print(f"Total symbols processed: {processed_count}")

# Call the function to compile the data
# compile_data()

def visualize_data():
    # Read the combined data from the CSV file
    file_path = 'nifty500_joined_closes.csv'
    df = pd.read_csv(file_path)
    df = df.drop(df.columns[0], axis=1)

    print(df)
    df_corr = df.corr()
    print(df_corr)

    data = df_corr.values
    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)
    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1, 1)
    
    plt.tight_layout()
    plt.show()

visualize_data()