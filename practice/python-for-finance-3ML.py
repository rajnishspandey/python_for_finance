import numpy as np
import pandas as pd 
from collections import Counter

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier




def process_data_for_labels(ticker):
    """
    This function takes a ticker symbol as input and returns the data for that ticker.
    """
    hm_days = 7  # Number of days to look back for data
    df = pd.read_csv('nifty500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)
    for i in range(1, hm_days + 1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
        # This line calculates the percentage change in the stock price for the next 'i' days
    df.fillna(0, inplace=True)
    return tickers, df
    # This line returns the list of tickers and the dataframe with the calculated percentage changes

# process_data_for_labels('TCS.NS')

def buy_sell_hold(*args):
    """
    This function takes a list of arguments representing the percentage changes for different time periods.
    It returns a string 'buy', 'sell', or 'hold' based on the conditions for buying, selling, or holding the stock.
    """
    cols = [c for c in args]
    requirement = 0.02  # Minimum requirement for buying or selling
    for col in cols:
        if col > requirement:
            return 'buy'
        if col < -requirement:
            return 'sell'
        return 'hold'
    
    # This line iterates over the list of percentage changes and returns 'buy' if any change is greater than the requirement,
    # 'sell' if any change is less than the negative of the requirement, and 'hold' if none of the conditions are met.

def extract_featuresets(ticker):
    """
    This function takes a ticker symbol as input and returns the feature sets and corresponding labels for that ticker.
    """
    tickers, df = process_data_for_labels(ticker)
    df['{}_taget'.format(ticker)] = list(map(buy_sell_hold,
                                              df['{}_1d'.format(ticker)],
                                              df['{}_2d'.format(ticker)],
                                              df['{}_3d'.format(ticker)],
                                              df['{}_4d'.format(ticker)],
                                              df['{}_5d'.format(ticker)],
                                              df['{}_6d'.format(ticker)],
                                              df['{}_7d'.format(ticker)]
                                              ))
    vals = df['{}_taget'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)

    x = df_vals.values
    y = df['{}_taget'.format(ticker)].values
    return x, y, df

# extract_featuresets('TCS.NS')

def do_ml(ticker):
    """
    This function takes a ticker symbol as input and performs machine learning on the data for that ticker.
    It returns the confidence score for the prediction.
    """
    X, y, df = extract_featuresets(ticker)

    # Impute missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Create the VotingClassifier
    clf = VotingClassifier([
        ('lsvc', LinearSVC(random_state=42)),
        ('knn', KNeighborsClassifier()),
        ('rfor', RandomForestClassifier(random_state=42))
    ])

    # Fit the classifier
    clf.fit(X_train, y_train)

    # Evaluate the classifier on the test data
    confidence = clf.score(X_test, y_test)
    print(f'Accuracy: {confidence:.4f}')

    # Make predictions on the test data
    predictions = clf.predict(X_test)
    print("Prediction spread:", Counter(predictions))

    return confidence

# Example usage
# result = do_ml('HDFCBANK.NS')
# print(f"Confidence score: {result:.4f}")

def do_ml_all():
    """
    This function reads symbols from nifty500_symbols.csv and performs ML on each symbol.
    It returns a dictionary of confidence scores for each symbol.
    """
    # Read the CSV file
    df = pd.read_csv('nifty500_symbols.csv')
    
    # Get the list of symbols
    symbols = df['symbol'].tolist()
    
    results = {}
    for symbol in symbols[:4]:
        try:
            confidence = do_ml(symbol)
            results[symbol] = confidence
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
    
    return results

# Run the analysis for all symbols
# all_results = do_ml_all()

# # Print overall results
# print("\nOverall Results:")
# for symbol, confidence in all_results.items():
#     print(f"{symbol}: {confidence:.4f}")

# # Calculate and print average confidence
# avg_confidence = np.mean(list(all_results.values()))
# print(f"\nAverage Confidence: {avg_confidence:.4f}") 

# # Identify symbols with confidence above a threshold
# threshold = 0.5  # Adjust this value as needed
# good_symbols = [symbol for symbol, confidence in all_results.items() if confidence > threshold]
# print("\nSymbols with confidence above", threshold, ":")
# for symbol in good_symbols:
#     print(symbol)
#     print("Confidence:", all_results[symbol])
