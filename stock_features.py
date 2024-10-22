import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def calculate_sma(data, window):
    """Calculate Simple Moving Average."""
    return data['close'].rolling(window=window).mean()

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index."""
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def generate_features(data):
    """Generate technical indicators and features for the model."""
    data['SMA20'] = calculate_sma(data, 20)
    data['SMA50'] = calculate_sma(data, 50)
    data['SMA20_Uptrend'] = (data['SMA20'] > data['SMA20'].shift(1)).astype(int)
    data['Returns'] = data['close'].pct_change(fill_method=None)
    data['RSI'] = calculate_rsi(data)
    return data

def create_labels(data, forward_days=5):
    """Create labels for future stock movement."""
    data['Future_Returns'] = data['close'].pct_change(periods=forward_days, fill_method=None).shift(-forward_days)
    data['Label'] = (data['Future_Returns'] > 0.02).astype(int)
    return data

def train_model(features, labels):
    """Train a Random Forest model and return it."""
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test
