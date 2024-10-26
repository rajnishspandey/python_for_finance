from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import logging
from stock_data import fetch_stock_data
from stock_features import generate_features, create_labels, train_model
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_stock(symbol):
    """Process stock data, generate features, train model, and make predictions."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        data = fetch_stock_data(symbol, start_date, end_date)
        data.set_index('date', inplace=True)
        data = data.sort_index(ascending=False)
        
        if data.empty:
            logging.warning(f"No data fetched for {symbol}.")
            return None

        data = generate_features(data)
        data = create_labels(data)
        data.dropna(inplace=True)

        if data.empty:
            logging.warning(f"No data after dropping NaNs for {symbol}.")
            return None
        
        features = data[['SMA20_Uptrend', 'Returns', 'RSI']]
        labels = data['Label']
        
        model, X_test, y_test = train_model(features, labels)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        
        # Check if data is already sorted
        if data.index[0] < data.index[-1]:
            data.sort_index(ascending=False, inplace=True)

        latest_data = features.iloc[-1:].copy()  # Always takes the last row for prediction
        logging.info(f"Latest Data for {symbol}: {latest_data}")

        
        # logging.info(f"Current Price: {current_price}, SMA20 Value: {sma20_value}")        
        
        # Make predictions for the next day
        prediction = model.predict(latest_data)[0]
        prediction_proba = model.predict_proba(latest_data)[0]

        # Define a threshold for "touching" the SMA20
        threshold = 0.01  # 1% proximity to SMA20

        # Check the conditions for storage
        current_price = data['close'].iloc[-1]
        sma20_value = data['SMA20'].iloc[-1]
        closing_date = data.index[-1]  # Assuming the index is the date
        
        if (latest_data['SMA20_Uptrend'].iloc[0] == 1 and 
            prediction == 1 and 
            current_price > sma20_value and 
            abs(current_price - sma20_value) / sma20_value <= threshold):

            return {
                'symbol': symbol,
                'Accuracy': round(accuracy, 2),
                'Prediction': 'Buy',
                'Confidence': prediction_proba[1],
                'Current_Price': current_price,
                'SMA20': sma20_value,
                'Is_Above_SMA20': current_price > sma20_value,
                'Distance_to_SMA20': abs(current_price - sma20_value),  # Optionally include distance
                'Closing_Date': closing_date.strftime('%Y-%m-%d')  # Format the date as a string
            }
        else:
            return None  # Return None if conditions are not met
    except Exception as e:
        logging.error(f"Error processing {symbol}: {str(e)}")
        return None

def predict_stocks(num_stocks=500, max_workers=None):
    """Predict stocks based on their historical data."""
    try:
        symbols_df = pd.read_csv('nifty500_symbols.csv')
        symbols_to_predict = symbols_df['symbol'].tolist()[:num_stocks]

        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(process_stock, symbol): symbol for symbol in symbols_to_predict}
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as exc:
                    logging.error(f"Stock {symbol} generated an exception: {exc}")

        # Convert results to DataFrame and filter based on conditions
        filtered_results = [result for result in results if result is not None]
        results_df = pd.DataFrame(filtered_results)

        return results_df

    except FileNotFoundError:
        logging.error("The file 'nifty500_symbols.csv' was not found.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        logging.error("The file 'nifty500_symbols.csv' is empty.")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    results_df = predict_stocks()
    if not results_df.empty:
        # Save results to an Excel file
        results_df.to_excel('predicted_stocks.xlsx', index=False)
        logging.info("Predicted stocks saved to 'predicted_stocks.xlsx'.")
    else:
        logging.warning("No results to save.")
