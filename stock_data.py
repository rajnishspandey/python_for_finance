from nsepython import nsefetch
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
from yahoo_fin import stock_info as si

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_nifty500_tickers():
    positions = nsefetch('https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500')
    df = pd.DataFrame(positions['data'])
    symbols = df[df['symbol'] != 'NIFTY 500']['symbol']
    symbols = symbols + '.NS'
    # Save to CSV
    # df.to_csv('nifty500_data.csv', index=False)
    symbols.to_csv('nifty500_symbols.csv', index=False, header=['symbol'])
    print("Data saved to nifty500_data.csv")

def get_data_from_yahoo():
    """Load the data from Yahoo Finance into CSV files, with each symbol in its own file."""
    # Read the symbols from the CSV file
    symbols_df = pd.read_csv('nifty500_symbols.csv')
    os.makedirs('stocks_dfs', exist_ok=True)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)

    for symbol in symbols_df['symbol']:
        try:
            yahoo_symbol = f"{symbol}"
            df = si.get_data(yahoo_symbol, start_date=start_date, end_date=end_date)
            df = df.sort_index(ascending=False)

            # Rename the index to 'date' and reset index
            df.index.name = 'date'
            df.reset_index(inplace=True)

            # Save the DataFrame to a CSV file
            file_path = os.path.join('stocks_dfs', f'{symbol}.csv')
            df.to_csv(file_path, index=False)
            print(f"Data for {symbol} saved to {file_path}")

        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")

# get_data_from_yahoo()


def fetch_stock_data(symbol, start_date, end_date):
    """Load stock data from a local CSV file and filter by date."""
    data_file = os.path.join('stocks_dfs', f'{symbol}.csv')
    
    try:
        data = pd.read_csv(data_file, header=0)

        # Clean column names
        data.columns = data.columns.str.strip()

        # Ensure 'date' is in datetime format
        data['date'] = pd.to_datetime(data['date'])
        
        # Filter data by date range
        data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        return data
    
    except Exception as e:
        logging.error(f"Error reading data for {symbol}: {str(e)}")
        return pd.DataFrame()
