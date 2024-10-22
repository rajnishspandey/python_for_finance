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
        start_date = end_date - timedelta(days=365)  # Fetch data for the past year
        data = fetch_stock_data(symbol, start_date, end_date)
        
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
        
        # Get the most recent data point
        latest_data = features.iloc[-1:].copy()
        
        # Make predictions for the next day
        prediction = model.predict(latest_data)[0]
        prediction_proba = model.predict_proba(latest_data)[0]

        return {
            'symbol': symbol,
            'Accuracy': round(accuracy,2),
            'Prediction': 'Buy' if prediction == 1 else 'Hold/Sell',
            'Confidence': prediction_proba[1] if prediction == 1 else prediction_proba[0],
            'Current_Price': data['close'].iloc[-1],
            'Closing_date': data['date'].iloc[-1],
            'SMA20': data['SMA20'].iloc[-1],
            'Is_Above_SMA20': data['close'].iloc[-1] > data['SMA20'].iloc[-1]
        }
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

        return pd.DataFrame(results)

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
